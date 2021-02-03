import pandas as pd

from src.scorers.utils import *
from tqdm import tqdm
from src.scorers.iim_scorer import ItemToItemScorer


class UserToItemScorer:

    def __init__(
            self, client, history_index, item_index, min_score, max_score, iim_mtx,
            uim_mtx, update_from, update_to, cfg, is_init=False
    ):
        self.client = client
        self.history_index = history_index
        self.item_index = item_index
        self.min_score = min_score
        self.max_score = max_score

        self.update_from = update_from
        self.update_to = update_to
        self.history_action_mapping_dict, self.history_action_rating_dict = self._get_mapping_tables()
        self.cfg = cfg
        self.unique_events_seen = set()

        self.iim_mtx = iim_mtx
        self.uim_mtx = uim_mtx

        # If these are not present, create them from scratch
        if is_init:
            self._init_iim_mtx()
            self.uim_mtx = self._init_uim_mtx(self.uim_mtx)

    def _init_iim_mtx(self):
        """
        Fall here if the item to item (non-user SoCaTel entity) relationships have not yet been formed. The output is
        saved in the accompanying MongoDB instance in the format "item_1::item_2::similarity_score"
        :return: None
        """
        total_items = get_index_max_id(self.client, self.item_index, self.cfg['schema'][self.item_index]['id'])

        if self.iim_mtx is None:
            iim_scorer = ItemToItemScorer(
                client=self.client, item_index=self.item_index, sec_item_index=None,
                update_from=self.update_from, update_to=self.update_to, cfg=self.cfg
            )

            iim = iim_scorer.get_item_to_item_matrix()

            # Save to MongoDB.
            iim_pio, iim_pio_non_zeroes, _ = pio_uim_transform(iim, is_iim=True)
            batch_mongo_insert_pio_data(iim_pio, self.cfg, self.item_index, self.item_index)
            batch_mongo_insert_pio_data(iim_pio_non_zeroes, self.cfg, self.item_index, self.item_index, non_zeroes=True)

            self.iim_mtx = iim
        elif len(self.iim_mtx) < total_items:
            # Add as many rows as columns for missing items.
            items_diff = total_items - len(self.iim_mtx)

            # Process columns.
            columns_df = pd.DataFrame(np.zeros(shape=(0, items_diff)))
            columns_df.columns = list(range(total_items - len(self.iim_mtx), total_items, 1))
            self.iim_mtx = pd.concat([self.iim_mtx, columns_df])  # No need to use axis=1 as `columns_df` is empty.

            # Process rows.
            rows_df = pd.DataFrame(np.zeros(shape=(items_diff, total_items)))
            rows_df.index = list(range(total_items - len(self.iim_mtx), total_items, 1))
            self.iim_mtx = pd.concat([self.iim_mtx, rows_df])
            self.iim_mtx.fillna(0)

    def _init_uim_mtx(self, uim_mtx):
        """
        Check whether the user to item matrix is ready for processing by checking its dimensions against total items.
        Initialise for the class accordingly.
        :param uim_mtx: (pandas.DataFrame) The current user to item matrix, if any. Must supply None if not.
        :return: None
        """
        total_items = get_index_max_id(self.client, self.item_index, self.cfg['schema'][self.item_index]['id'])

        if uim_mtx is None:
            uim_mtx = pd.DataFrame(columns=list(range(total_items)))
        elif len(uim_mtx.columns) < total_items:
            len_diff = total_items - len(uim_mtx.columns)

            # Only check if there are less. We have no way of tracking deleted services, so keep a reference in the
            # matrix. Expand with zero array vectors if there more.
            added_df = pd.DataFrame(
                columns=list(range(len(uim_mtx.columns), len(uim_mtx.columns) + len_diff)))
            uim_mtx = pd.concat([uim_mtx, added_df])
            uim_mtx.fillna(0, inplace=True)

        return uim_mtx

    @staticmethod
    def _get_mapping_tables():
        """
        Get scoring scheme for user actions
        :return: (dict) Key-value mapping user actions to scores
        """
        # Only create a mapping on actions of interest.
        history_action_mapping_dict = {
            "3": "CREATED_GROUP",
            "4": "POSTED_GROUP",
            "5": "SUBSCRIBED_GROUP",
            # Keep a record of groups they unsubscribe from as a separate recommendation, after some time passes?
            "6": "UNSUBSCRIBED_GROUP",
            "7": "VIEWED_GROUP",
            "27": "MESSAGE_USER",
            "26": "SEARCH_TOPIC"
        }

        # Maintain ratings between 0-100 and prune over-under ratings. Values should match `history_action_rating_dict`
        history_action_rating_dict = {
            "history.created_group": 100,  # 3
            "history.posted_group": 10,  # 4
            "history.subscribed_group": 15,  # 5
            "history.unsubscribed_group": -20,  # 6
            "history.viewed_group": 5,  # 7
            "history.joined_group": 25
        }

        return history_action_mapping_dict, history_action_rating_dict

    def update_from_latest_history_posts(self):
        """
        User to item matrix update based on a set time interval.
        :return: The number of activity entries processed.
        """
        new_users = list()
        items_size = get_index_max_id(self.client, self.item_index, self.cfg['schema'][self.item_index]['id'])

        if items_size > len(self.uim_mtx.columns):
            items_diff = items_size - len(self.uim_mtx.columns)

            columns_df = pd.DataFrame(np.zeros(shape=(0, items_diff)))
            columns_df.columns = list(range(items_size - items_diff, items_size, 1))
            self.uim_mtx = pd.concat([self.uim_mtx, columns_df])  # No need to use axis=1 as `columns_df` is empty.
            self.uim_mtx.fillna(0, inplace=True)

        # Given these do not exceed the result window size, set at 10,000
        history_list = self.client.search(index=self.history_index,
                                          body={
                                              "size": 10000,
                                              "query": {
                                                  "bool": {
                                                      "must": [
                                                          {
                                                              "range": {
                                                                  "history_timestamp": {
                                                                      "gte": self.update_from,
                                                                      "lt": self.update_to
                                                                  }
                                                              }
                                                          }
                                                      ]
                                                  }
                                              }
                                          }
                                          )

        history_hits = history_list['hits']['hits']

        # More services might have been added since the last update. Add new columns in the dataset to represent these
        # new services.
        interacted_with = list()
        for hit in history_hits:
            if hit['_source']['user_id'] not in list(self.uim_mtx.index):
                new_users.append(hit['_source']['user_id'])
                self.uim_mtx.loc[hit['_source']['user_id']] = np.zeros(shape=(len(self.uim_mtx.columns)))

            self.uim_mtx.loc[hit['_source']['user_id']] = self._update_user_item_mtx(
                hit['_source'][self.cfg['schema'][self.item_index]['id']], hit['_source']['history_text'],
                self.uim_mtx.loc[hit['_source']['user_id']]
            )

            interacted_with.append([hit['_source']['user_id'],
                                    hit['_source'][self.cfg['schema'][self.item_index]['id']] - 1])

        # Add one to negate decay for groups interacted with today.
        for interaction in interacted_with:
            self.uim_mtx.loc[interaction[0], interaction[1]] += 1

        self.uim_mtx -= 1
        # Correct negative scores
        self.uim_mtx[self.uim_mtx < 0] += 1

        # Only calculate similar scores for this fraction of users
        self.uim_mtx.loc[new_users] = self._score_similar_items_if_null(self.uim_mtx.loc[new_users])
        self.uim_mtx.loc[new_users] = self.uim_mtx.loc[new_users].apply(normalize, axis=1)

        return len(history_hits)

    def _prune_user_score(self, score):
        """
        User score should be between min_n and max_n
        :param score: (int) current score after update based on user action
        :return: (int) pruned score within limits.
        """
        return max(min(self.max_score, score), self.min_score)

    def _update_user_item_mtx(self, service_idx, user_action, user_vector):
        """
        Given a user vector, update the corresponding service rating based on a given user action
        :param service_idx:
        :param user_action: (str) The name of the user's action
        :param user_vector: (np.array) The user's service scores
        :return: (np.array) The updated user vector with item scores
        """
        # Update action
        if user_action == 'history.message_user':
            # Get all services which the user messaged to participates and add a point to the current user's vector in
            # corresponding service scores.
            pass

        if user_action not in self.history_action_rating_dict.keys():
            # No action was made that had an impact to a service's score. Return.
            return user_vector

        # Map to array elements.
        item_idx = service_idx - 1

        try:
            # The score will never be zero for items that have been actioned upon at least once.
            # Min score for pruning should always be > 0
            user_vector[item_idx] = self._prune_user_score(
                float(user_vector[item_idx]) + self.history_action_rating_dict[user_action]
            )
        except KeyError:
            # There seem to be some groups that were deleted, but the history associated with them is still there.
            # Do not record scores for those that were deleted.
            pass

        return user_vector

    def _de_sparsify_uim_by_iim(self, uim_vector, sim_thresh=0.55):
        """
        **Use pd.apply() on this.
        This function is meant to populate with a rating entries in the user to item matrix which do not have one, given
        that we have access to similar items to the ones missing and, for those similar items, we have at least one that
        has a high enough score rated by the user.
        :param uim_vector: A user's ratings to items as a vector
        :param sim_thresh: The minimum item to item similarity score to be considered.
        :return: (pd.Series) The updated user to items vector.
        """
        for item in list(uim_vector.index):  # use the index (previously column names) to query the item to item matrix.
            item = int(item)
            # Get iim_vector
            iim_vector = self.iim_mtx.iloc[item]

            counter = 1
            while counter < len(iim_vector):
                # Get next largest similarity score.
                sorted_iim = iim_vector.sort_values(ascending=False).reset_index(drop=False)  # We need the original
                sorted_iim.columns = ['original_index', 'value']

                sim_item_val = sorted_iim.loc[counter, 'value']
                sim_item_idx = int(sorted_iim.loc[counter, 'original_index'])
                counter += 1

                if sim_item_val < sim_thresh:
                    # Reached an item which does not have a high enough threshold. Since these are retrieved in a
                    # descending fashion, it is safe to stop and return the vector as is here. Go to next item.
                    break

                if uim_vector.iloc[item] == 0 or uim_vector.iloc[sim_item_idx] != 0:
                    # Score is not available for this similar item.
                    # This could also occur  when the score given by a user for the similar one is not zero,
                    # meaning that it has already been rated.
                    # The score could originate either by the user or by a previous run.
                    continue

                # Set the rating of an item, which was not rated by the user, to be the rating the user gave to the
                # most similar item to the non-rated one times the similarity weight found in the item to item matrix.
                uim_vector.iloc[sim_item_idx] = uim_vector.iloc[item] * sim_item_val

        return uim_vector

    def _score_similar_items_if_null(self, user_item_mtx):
        """
        Use the similarity score between items given by elasticsearch as a weight to the overall score of items similar
         to ones already scored.
        :param user_item_mtx: (pd.DataFrame) The current user to item matrix
        :return: (pd.DataFrame) Update user to item matrix to include scores for similar items to the ones present in
        the out of date matrix
        """
        user_ids = list(user_item_mtx.index)
        user_item_mtx.reset_index(inplace=True, drop=True)
        print("\nScoring similar items given ES Similarity API scores as weights\n")
        tqdm.pandas()
        # Can be slow. Takes ~5 minutes to return with a matrix of shape(133, 116)
        user_item_mtx = user_item_mtx.progress_apply(self._de_sparsify_uim_by_iim, axis=1)
        user_item_mtx.index = user_ids

        return user_item_mtx

    def update_from_all_history_posts(self):
        """
        In case all history posts are required to initialize the user to item matrix, then use this function. It
         utilizes the scroll API to stream all content in the index linearly.
        :return: (int) How many user activity entries have been processed so far.
        """
        # The user matrix should not exist if this is called.
        total_items = get_index_max_id(self.client, self.item_index, self.cfg['schema'][self.item_index]['id'])
        # get_index_max_id(self.client, self.history_index, self.cfg['schema'][self.history_index]['id'])
        total_history_posts = get_total_items_es(self.client, self.history_index)

        # declare a filter query dict object
        match_all = {
            "size": 100,
            "query": {
                "match_all": {}
            }
        }

        # make a search() request to get all docs in the index
        resp = self.client.search(
            index=self.history_index,
            body=match_all,
            scroll='10m'  # length of time to keep search context
        )

        del match_all

        # keep track of past scroll _id
        old_scroll_id = resp['_scroll_id']

        hits_counter = 0
        print("\nParsing through all items.\n")

        t = tqdm(total=total_history_posts)
        while len(resp['hits']['hits']):
            # iterate over the document hits for each 'scroll'
            for source in resp['hits']['hits']:
                # Create a new user vector
                doc = source['_source']

                if doc['user_id'] is None:
                    continue

                self.uim_mtx.loc[doc['user_id']] = np.zeros(shape=total_items)

                if doc['history_text'].startswith('history.'):  # Otherwise, it is not a known event
                    # Assign points to service ratings based on user actions
                    self.uim_mtx.loc[doc['user_id']] = self._update_user_item_mtx(
                        doc['group_id'], doc['history_text'], self.uim_mtx.loc[doc['user_id']]
                    )

                hits_counter += 1
                t.update(1)

            # Make a new request using the Scroll API
            resp = self.client.scroll(
                scroll_id=old_scroll_id,
                scroll='10m'
            )

            # check if there's a new scroll ID
            if old_scroll_id != resp['_scroll_id']:
                print("NEW SCROLL ID:", resp['_scroll_id'])

            # keep track of past scroll _id
            old_scroll_id = resp['_scroll_id']

        t.close()

        # Since there is very sparse user activity, we need to stimulate scoring for active users on similar
        # groups to the one they participate in. This needs to happen when we have full picture of the user's activity.
        self.uim_mtx = self._score_similar_items_if_null(self.uim_mtx)

        return hits_counter
