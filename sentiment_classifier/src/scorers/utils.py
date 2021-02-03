import re
import nltk
import string
import pymongo
import requests
import argparse
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo.errors import WriteError

LANG_MAPPING = {
    "1": "english",
    "2": "spanish",
    "3": "finnish",
    "4": "hungarian",
    # Although this is Catalan, there are no libraries used here that support it. Use Spanish instead.
    "5": "spanish",
    "8": "italian"
}

LANG_SHORT_MAPPING = {
    "1": "en",
    "2": "es",
    "3": "fi",
    "4": "hu",
    "5": "es",
    "8": "it"
}


def is_valid_word(word):
    """
    Check if word begins with an alphabet.
    :param word: (str) Single word
    :return: True if valid word, False otherwise
    """
    return re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None


def handle_stopwords(word, lang):
    """
    Check if a given word is a stop word, given the language it is in.
    :param word: (str) Given word
    :param lang: (str) Language code
    :return: (str) Empty if stopword, word if not
    """
    try:
        cur_stopwords = set(nltk.corpus.stopwords.words(lang))
    except LookupError:
        nltk.download('stopwords')
        cur_stopwords = set(nltk.corpus.stopwords.words(lang))

    if word not in cur_stopwords:
        return word
    else:
        return ""


def pre_process_free_text(text, lang_id, rm_stopwords=True, rm_keywords=list()):
    """
    Given a string, which can be a tweet or a service description here, clean it from:
    1. Pre-selected keywords
    2. Stopwords
    3. Special characters
    4. Character repetition
    5. Numeric characters
    :param text: (str) A block of text
    :param lang_id: (int) The id of the language within the SoCaTel specs
    :param rm_stopwords: (bool) Whether stopwords should be removed or not
    :param rm_keywords: (list) A list of keywords to be removed
    :return:
    """
    processed_text = list()
    text = text.lower()
    words = text.split()
    lang = LANG_MAPPING[str(lang_id)]

    for word in words:
        # Listed as a keyword not to include.
        if word in rm_keywords:
            continue

        word = pre_process_word(word)
        if rm_stopwords:
            word = handle_stopwords(word, lang)
        if is_valid_word(word):
            processed_text.append(word)

    return processed_text


def pre_process_word(word):
    """
    Clean words from special characters and potential typos.
    :param word: (str) Unprocessed word
    :return: (str) Preprocessed word
    """
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter, e.g. funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & ' & #
    word = re.sub(r'(-|\'|#)', '', word)
    return word


def str2bool(v):
    """
    Useful for command line arguments, this function returns the respective boolean based on the string input
    :param v: (string) The string input indicating a boolean value
    :return: (bool) The boolean value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def normalize(x):
    """
    Normalize vector in a 0-1 scale.
    :param x: (np.array) An array with numerical values
    :return: (np.array) Normalised array
    """
    if x.max() > 0:
        return (x-x.min())/(x.max()-x.min())
    else:
        return x


def get_total_items_es(client, idx):
    """
    Get total documents in an index.
    :param client: Elasticsearch client
    :param idx: (str) The name of the index
    :return: (int) Total documents
    """
    total_docs = client.indices.stats(idx)['_all']['primaries']['docs']['count']
    return total_docs


def get_index_max_id(client, idx, id_name):
    """
    Get the maximum id of the document.
    This is useful in cases where the size of the index does not align with the maximum id recorded in the dataset
    (there are gaps in the id range).
    :param client: (elasticsearch.Elasticsearch) The connection object to elasticsearch
    :param idx: (str) The name of the Elasticsearch index to query
    :param id_name: (str) The name of the id holding field within the index to be queried
    :return: (int) The last/largest id of the Elasticsearch index
    """
    es_query = {
        "size": 0,
        "aggs": {
            "max_id": {
                "top_hits": {
                    "sort": [
                        {
                            id_name: {
                                "order": "desc"
                            }
                        }
                    ],
                    "size": 1
                }
            }
        }
    }

    res = client.search(index=idx, body=es_query)
    max_id = res['aggregations']['max_id']['hits']['hits'][0]['_id']
    return int(max_id)


def get_mongo_resources(cfg, primary_idx, secondary_idx=None):
    """
    Based on the configuration file provided, return connection objects to the Mongo database which corresponds to the
    Elasticsearch indices that are going to be queried.
    :param cfg: (dict) Configuration file, that includes connection details
    :param primary_idx: (str) The Elasticsearch index name we are querying.
    :param secondary_idx: (str) If a secondary Elasticsearch index is used, to create similarity scores between
    different types of items  for example.
    :return: (pymongo.MongoClient) Connection object
    """
    client = pymongo.MongoClient(host=cfg['mongo']['host'],
                                 port=cfg['mongo']['port'],
                                 username=cfg['mongo']['username'],
                                 password=cfg['mongo']['password']
                                 )

    if secondary_idx is None:
        db = client[cfg['mongo']['user'][primary_idx]['db_name']]
    else:
        db = client[cfg['mongo'][primary_idx][secondary_idx]['db_name']]

    return db, client


def get_score_mtx_from_mongo(cfg, primary_idx, is_iim, secondary_idx=None):
    """
    Return contents of all documents stored in the target Mongo collection, specified by the names of Elasticsearch
    indices that data originate from.
    :param cfg:  (dict) Configuration file, that includes connection details
    :param primary_idx: (str) The Elasticsearch index name we are querying.
    :param is_iim: (boolean) Flag to indicate whether this matrix holds non-user similarity information or not.
    :param secondary_idx: (str) If a secondary Elasticsearch index is used, to create similarity scores between
    :return: (pandas.DataFrame) 2-D dataframe of either user to item rating or item to item similarity scores
    """
    db, client = get_mongo_resources(cfg, primary_idx, secondary_idx)

    documents = []
    # Get the full spatial matrix (with zeroes)
    coll_name = [x for x in db.collection_names() if not x.endswith("non_zeroes")][0]

    collection = db[coll_name]
    cursor = collection.find({})
    for document in cursor:
        documents.append(document['combo'])

    client.close()

    return pio_format_to_pandas(documents, is_iim=is_iim)


def check_if_init_job(cfg, item_index, sec_item_index=None):
    """
    Check whether a relationship between SoCaTel entities has been previously recorded within MongoDB
    :param cfg: (dict) The dictionary containing configurations, including MongoDB connection details
    :param item_index: (str) Name of the SoCaTel entity from which we will use details of towards recommendations
    :param sec_item_index: (str) Name of the SoCaTel entity from which we will draw recommendations from. If None,
    this will be set as the value of `item_index`
    :return: (bool) Whether the relationship exists (True) or not (False)
    """
    db, client = get_mongo_resources(cfg, item_index, sec_item_index)

    res = len(db.collection_names()) == 0
    client.close()
    return res


def run_query(query, cfg):
    """
    API call to EVERIS' GraphQL endpoints
    :param query: (str) Fully formed GraphQL query to be sent over as request body
    :param cfg: (dict) Configuration file, that includes connection details
    :return:
    """
    headers = {
        'Authorization': cfg['graphql']['graphQL_authorisation']
    }

    request = requests.post(cfg['graphql']['graphQL_endpoint'], json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        return None


def graph_ql_tw_search_by_topics(topics, lang_short, cfg):
    """
    Given a list of topics/keywords, use EVERIS' GraphQL endpoint to search for relevant tweets to those topics/keywords
    :param topics: (list) List of topics/keywords
    :param lang_short: The shorthand of the language of the tweets to search for
    :param cfg: (dict) Configuration file, that includes connection details
    :return:
    """
    if not isinstance(topics, list):
        print('Please provide a list of topics.')
        return None

    query = cfg['graphql']['graphql_twSearchByTopics']
    query = query.replace("$language", lang_short)
    # Need double quotes for GraphQL call
    query = query.replace("$topics", str(topics).replace("'", '"'))

    try:
        return run_query(query, cfg)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError):
        return None


def get_frequent_words_across_tweets(topics, lang_id, client, cfg, top_words_size=10):
    """
    Get the GraphQL tweet entries output and get the most frequent words across tweets
    :param topics: (list) List of topics/keywords
    :param lang_id: (int) The id of the language within the SoCaTel specs
    :param client: (pymongo.MongoClient)
    :param cfg: (dict) Configuration file, that includes connection details
    :param top_words_size: How many words, ranked by popularity in tweets found, should be returned.
    :return:
    """
    lang_short = LANG_SHORT_MAPPING[str(lang_id)]

    dict_res = graph_ql_tw_search_by_topics(topics, lang_short, cfg)

    if dict_res is None:
        return []

    if len(dict_res['data']['postsByTopics']) == 0:
        return []

    tweet_ids = list(set([tweet['identifier'] for tweet in dict_res['data']['postsByTopics']]))
    # Now query ES for all relevant tweets
    tweets = client.search(
        index=cfg['elasticsearch']['twitter_index'],
        body={
            "_source": ["text"],
            "query": {
                "bool": {
                    "filter": {
                        "terms": {
                            "id_str": tweet_ids
                        }
                    }
                }
            }
        }
    )

    if len(tweets['hits']['hits']) > 0:
        tweets_txt = [' '.join(pre_process_free_text(tweet['_source']['text'], lang_id))
                      for tweet in tweets['hits']['hits']]
        tf_idf = TfidfVectorizer(analyzer=lambda x: x.split(' ')).fit(tweets_txt)

        return tf_idf.get_feature_names()[0:top_words_size-1]
    else:
        return []


def batch_mongo_insert_pio_data(data, cfg, primary_idx, secondary_idx=None, non_zeroes=False):
    """
    Given a list containing data formatted to be read within PredictionIO, insert to the specified Mongo collection
    (indicated by the Elasticsearch indices' names).
    :param data: (list)
    :param cfg: (dict) Configuration file, that includes connection details
    :param primary_idx: (str) The Elasticsearch index name we are querying
    :param secondary_idx: (str) If a secondary Elasticsearch index is used, to create similarity scores between
    :param non_zeroes: (bool) Whether to store entries that have no user rating with a zero or not
    :return: True if insertion operation succeeds, False otherwise
    """
    db, client = get_mongo_resources(cfg, primary_idx, secondary_idx)

    # Only a single collection per use case. If it already exists, overwrite with new one.
    if secondary_idx is None:
        coll_name = 'user_to_' + str(primary_idx)
    else:
        coll_name = str(primary_idx) + '_to_' + str(secondary_idx)

    if non_zeroes:
        coll_name += "_non_zeroes"

    if coll_name in db.collection_names():
        db[coll_name].drop()

    res = True
    try:
        db[coll_name].insert_many(data)
    except WriteError as we:
        print(we)
        res = False

    client.close()
    return res


def pio_format_to_pandas(mongo_docs, col_names=None, is_iim=False):
    """
    This function accepts documents from a MongoDB collection and converts them to a pandas.DataFrame()
    :param mongo_docs: The collection documents within a list of dictionaries
    :param col_names: A list of size 3, first element being the index name, second the column names
    and third the value names
    :param is_iim: (boolean) Identifier to show whether the dataframe to be retrieved containers relationships between
    items and NOT users to items
    :return: (pandas.DataFrame) The reconstructed dataframe, as firstly compiled through the handlers in this project
    """
    if col_names is None:
        col_names = ['X', 'Y', 'Z']

    if len(col_names) == 3:
        df = pd.DataFrame([x.split('::') for x in mongo_docs], columns=col_names)
        df = df[~df.duplicated(keep='first')].copy()
        df_ = df.pivot(index=col_names[0], columns=col_names[1], values=col_names[2])
        sorted_cols = list(df_.columns)
        sorted_cols.sort(key=int)

        df_ = df_[sorted_cols]
        df_.columns = [int(x) - 1 for x in list(df_.columns)]
        df_ = df_.sort_index(axis=1)

        # If item to item, both index and columns were saved with 1 as the starting index.
        # Revert so that we can leverage positioning access correctly
        if is_iim:
            df_.index = [int(x) - 1 for x in list(df_.index)]
            df_ = df_.sort_index(axis=0)

        return df_.apply(pd.to_numeric)
    else:
        return None


def apply_pio_format(row, is_iim=False, write_zeroes=False):
    """
    Pio recommendation engine requires data in "user::item:rating" format. Given these data in matrix format, convert
    them accordingly
    :param row: (pd.Series) Applied on a pd.DataFrame, so each iteration requires a pd.Series.
    :param is_iim: (boolean)
    :param write_zeroes: (boolean) Whether relationships not established yet (marked with a score of zero) should also
    be written out to the output
    :return: (pd.Series) The converted scores
    """
    row_scores = list()
    for col in list(row.index):
        col_val = int(col)
        row_val = row.name
        col_val += 1

        if is_iim:
            row_val += 1

        if not write_zeroes and float(row.loc[col]) == 0:
            continue

        row_scores.append('{}::{}::{}'.format(row_val, col_val, row.loc[col]))

    if len(row_scores) == 0:
        return None

    return row_scores


def expand_iim_if_necessary(es, iim_mtx, item_index, sec_item_index, cfg):
    """
    Identify whether new items exist in the platform and expand the dataframe stored in MongoDB to accommodate them.
    :param es: Elasticsearch connection buffer
    :param iim_mtx: (pandas.DataFrame) The item to item matrix in dataframe format.
    :param item_index: (str) The name of the Elasticsearch index hosting the item type in question.
    :param sec_item_index: (str) If the relationship concerns one item to a different one, supply it here. Can be None
    :param cfg: (dict) Configuration file, that includes connection details
    :return: (pandas.DataFrame) The possibly expanded item to item matrix
             (boolean) Whether the dataframe was expanded or not.
    """
    total_items_rows = get_index_max_id(es, item_index, cfg['schema'][item_index]['id'])
    if sec_item_index is not None:
        total_items_cols = get_index_max_id(es, sec_item_index, cfg['schema'][sec_item_index]['id'])
    else:
        total_items_cols = total_items_rows

    needs_expansion = len(iim_mtx) < total_items_rows
    if needs_expansion:
        # Add as many rows as columns for missing items.
        items_diff_rows = total_items_rows - len(iim_mtx)
        items_diff_cols = total_items_cols - len(iim_mtx.columns)

        # This can only be false if there is a secondary item used, but can still be true for secondary items
        if len(iim_mtx.columns) < total_items_cols:
            # Process columns
            columns_df = pd.DataFrame(np.zeros(shape=(0, items_diff_cols)))
            columns_df.columns = list(range(len(iim_mtx.columns), total_items_cols, 1))
            iim_mtx = pd.concat([iim_mtx, columns_df], axis=1)  # No need to use axis=1 as `columns_df` is empty.

        # Process rows
        rows_df = pd.DataFrame(np.zeros(shape=(items_diff_rows, 0)))
        rows_df.index = list(range(len(iim_mtx), total_items_rows, 1))
        iim_mtx = pd.concat([iim_mtx, rows_df])
        iim_mtx.fillna(0, inplace=True)

    return iim_mtx, needs_expansion


def pio_uim_transform(mtx, is_iim, as_str=False):
    """
    Transform a dataframe into a list of dictionaries, of which values are in the format that PIO requires them in.
    :param mtx: (pandas.DataFrame) 2-D dataframe representing ratings between two entities (user, SoCaTel items, etc.)
    :param is_iim: (pandas.DataFrame) Identifier to show whether the dataframe to be retrieved containers
    relationships between
    :param as_str: (boolean) Whether to construct a String object containing the scores
    :return: (list of dicts) PIO formatted list of dictionaries - all data
             (list of dicts) PIO formatted list of dictionaries - excluding relationships that are zero
             (str) PIO formatted string with relationships
    """
    pio_mtx = mtx.apply(apply_pio_format, is_iim=is_iim, write_zeroes=True, axis=1)
    pio_mtx_non_zeroes = mtx.apply(apply_pio_format, is_iim=is_iim, write_zeroes=False, axis=1).dropna()

    pio_mtx = list(pio_mtx.to_dict().values())
    pio_mtx = [{"combo": val} for sublist in pio_mtx
               if sublist is not None for val in sublist if val is not None]

    pio_mtx_non_zeroes = list(pio_mtx_non_zeroes.to_dict().values())

    pio_mtx_non_zeroes_str = ""
    if as_str:
        try:
            pio_mtx_non_zeroes_str = "\n".join([item for sublist in pio_mtx_non_zeroes for item in sublist])
        except TypeError:
            pass

    pio_mtx_non_zeroes = [{"combo": val} for sublist in pio_mtx_non_zeroes
                          if sublist is not None for val in sublist if val is not None]

    return pio_mtx, pio_mtx_non_zeroes, pio_mtx_non_zeroes_str
