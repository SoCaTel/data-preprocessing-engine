import elasticsearch_dsl as edsl

from src.scorers.utils import *
from tqdm import tqdm


class ItemToItemScorer:
    """
    Holds helper functions for creating and storing item to item matrices for use in recommendation engines. `Item` and
    `Group` are used inter-changeably here.
    """
    def __init__(self, client, item_index, sec_item_index, update_from, update_to, cfg):
        self.client = client
        self.item_index = item_index
        self.sec_item_index = sec_item_index
        self.update_from = update_from
        self.update_to = update_to
        self.cfg = cfg

    def get_item_to_item_matrix(self):
        """
        Construct item to item similarity matrix, from items in the SoCaTel elasticsearch group index
        :return: (pandas.DataFrame) The item to item matrix in pandas format. Index and column names serve as the group
        ids found in the Elasticsearch index.
        """
        total_items_main = get_index_max_id(self.client, self.item_index, self.cfg['schema'][self.item_index]['id'])

        if self.sec_item_index is None:
            item_item_mtx = pd.DataFrame(columns=list(range(total_items_main)))
        else:
            total_items_sec = get_index_max_id(self.client, self.sec_item_index,
                                               self.cfg['schema'][self.sec_item_index]['id'])
            item_item_mtx = pd.DataFrame(np.zeros(shape=(total_items_main, total_items_sec)))

        return self._construct_item_group_similarity_query(item_item_mtx, self.item_index, self.sec_item_index)

    def update_from_latest_history_posts(self, iim_matrix):
        """
        Update an existing item to item matrix by querying the index on a set time period for new items inserted.
        :param iim_matrix: (pandas.DataFrame) Out of date item to item matrix
        :return: (pandas.DataFrame) Updated item to item matrix
        """
        new_items_q = self.client.search(
            index=self.item_index,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    self.cfg[self.item_index]['create_time']: {
                                        "gte": self.update_from,
                                        "lt": self.update_to,
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        )

        new_items = new_items_q['hits']['hits']
        new_item_ids = list()
        print("\nParsing through new items.\n")
        for item in tqdm(new_items):
            # Init to zeroes
            item_id = item[self.item_index]['id']
            new_item_ids.append(item_id)
            iim_matrix.loc[item_id - 1] = np.zeros(shape=len(iim_matrix.columns))

        # Apply this only on the fraction of item ids that will need update.
        iim_matrix.loc[new_item_ids] = self._construct_item_group_similarity_query(
            iim_matrix.loc[new_item_ids], self.item_index, self.sec_item_index
        )

        return iim_matrix

    def _get_item_similarity_criteria(
            self, idx, item_lang_id, item_themes, keywords, item_locality_id, item_locality_parent
    ):
        """
        Construct an Elasticsearch item similarity query.
        :param idx: (str) The name of the Elasticsearch index to query
        :param item_lang_id: (str) The field holding the language id for the specified Elasticsearch index.
        :param item_themes: (list) A list of strings that hold the themes for which to filter results
        :param keywords: (list) A list of strings holding any extra keywords to be considered while ranking results
        retrieved
        :param item_locality_id: (str) The field holding the locality id for the specified Elasticsearch index.
        :param item_locality_parent: (str) The field holding the locality parent id for the specified Elasticsearch
        index.
        :return: (list, list) Two lists holding the (a) query with which to filter results and (b) the query with
        which to rank results (must and should elasticsearch query clauses respectively).
        """
        must_list = list()
        should_list = list()

        if 'language_id' in self.cfg['schema'][idx]:
            lang_code_schema = self.cfg['schema'][idx]['language_id']

            must_list.append({
                'match_phrase': {
                    lang_code_schema: item_lang_id
                }
            })

        if 'theme_name' in self.cfg['schema'][idx]:
            themes_name_schema = self.cfg['schema'][idx]['theme_name']
            themes_name_root_term = self.cfg['schema'][idx]['theme_name'].split(".")[-1]

            # Issue a similarity query and issue scores of returned services in the item vector
            [
                should_list.append({
                    'match_phrase': {
                        themes_name_schema: theme[themes_name_root_term]
                    }
                }) for theme in item_themes
            ]

            item_theme_names = [x["theme_name"] for x in item_themes]
            extra_keywords = get_frequent_words_across_tweets(item_theme_names, item_lang_id, self.client, self.cfg)

            if len(extra_keywords) == 0:  # got nothing
                extra_keywords = get_frequent_words_across_tweets(keywords, item_lang_id, self.client, self.cfg)

            keywords.extend(extra_keywords)

        # Fuzzy match on the title only. Fuzzy matching on bigger strings (such as the description) is costly
        for word in keywords:
            should_list.append({
                'fuzzy': {
                    self.cfg['schema'][self.item_index]["name"]: {
                        'value': word,
                        # Boost, as it would be important to find any keyword in the title of other services than
                        # matching anything else in the should clause (either locality or even finding keywords in
                        # the description).
                        'boost': 2,
                        'prefix_length': 1
                    }
                }
            })

            should_list.append({
                'fuzzy': {
                    self.cfg['schema'][self.item_index]["description"]: {
                        'value': word,
                        # Don't boost, as words in the description might hold less importance than in the title
                        'prefix_length': 1
                    }
                }
            })

            if "native_description" in self.cfg['schema'][self.item_index]:
                should_list.append({
                    'fuzzy': {
                        self.cfg['schema'][self.item_index]["native_description"]: {
                            'value': word,
                            # Don't boost, as words in the description might hold less importance than in the title
                            'prefix_length': 1
                        }
                    }
                })

        if 'locality_id' in self.cfg['schema'][idx]:
            locality_id_term = self.cfg['schema'][idx]['locality_id']
            locality_parent_id_term = self.cfg['schema'][idx]['locality_parent_id']

            should_list.append({'match': {locality_id_term: item_locality_id}})
            if item_locality_parent is not None:
                should_list.append({'match': {
                    locality_parent_id_term: item_locality_parent['locality_id']
                }})

        return must_list, should_list

    def _construct_item_group_similarity_query(self, iim, idx, sec_idx=None):
        """
        **Same item type similarity scorer**
        This is a process that calls upon an Elasticsearch query which can produce a number of similar documents,
        given the following item (SoCaTel group) attributes:
        1. Language
        2. Themes
        3. Title and item descriptions (using fuzzy matching)
        4. Item locality (where the group in the world is taking place)
        :param iim: (pandas.DataFrame) The given dataframe needs to have its index and columns point to the
        item ids to be queried in the given Elasticsearch index (`idx`)
        :param idx: (str) Group index is typically used, but we generalise this function to work on any "item" holding
        index
        :return: (pandas.DataFrame) Updated item to item matrix with similarity scores.
        """
        for item in list(range(0, len(iim.columns))):
            # Init for every item
            item += 1  # Ids in elasticsearch start with 1
            # Get info on the item in question
            item_info_q = edsl.Search(index=idx).using(client=self.client).query("match", **{"_id": item})
            item_info = item_info_q.execute()

            # Should only return one result as id is unique
            if len(item_info) == 1:
                item_info_dict = item_info[0].to_dict()
                item_themes = item_info_dict['themes']
                item_lang_id = item_info_dict['language']['language_id']

                item_title = item_info_dict[self.cfg['schema'][idx]['name']]
                title_keywords = pre_process_free_text(item_title, item_lang_id)

                item_locality_id = item_info_dict['locality']['locality_id']
                item_locality_parent = item_info_dict['locality']['locality_parent']
            else:
                # Either item does not exist or there is an Elasticsearch indexing error. Set all to zero to avoid
                # further indexing errors.
                iim.loc[item - 1] = np.zeros(shape=len(iim.columns))
                continue

            # We got the item info from the main index. If we are to get similarity scores with another "item" holding
            # index, then we need to create and issue queries on the alternative index given.
            idx_to_query = idx
            if sec_idx is not None:
                idx_to_query = sec_idx

            must_list, should_list = self._get_item_similarity_criteria(
                idx_to_query, item_lang_id, item_themes, title_keywords, item_locality_id, item_locality_parent
            )

            s = edsl.Search(index=idx_to_query).using(client=self.client).query(
                "bool",
                **{"must": must_list,
                   "should": should_list,
                   "minimum_should_match": 1
                   })

            similar_items = s.execute()
            iim.loc[item - 1] = np.zeros(shape=len(iim.columns))
            sim_hits = similar_items.to_dict()['hits']['hits']

            if len(sim_hits) > 0:
                sim_df = pd.DataFrame(similar_items.to_dict()['hits']['hits'])[['_id', '_score']]
                sim_ids = [int(x) - 1 for x in sim_df['_id'].to_list()]  # Map to 0-starting index
                iim.loc[item - 1, sim_ids] += sim_df['_score'].to_list()

                # Normalize the vector to 0 - 1. Don't normalize for the whole matrix, as score scales from item to item
                # vector are more abstractly different.
                iim.loc[item - 1] = normalize(iim.loc[item - 1])

        return iim
