import yaml
import argparse

from tqdm import tqdm
from src.scorers.utils import *
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta
from urllib.parse import quote_plus as urlquote
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_groups_by_lang(es_client, language_code, group_index='so_group'):
    """
    Retrieve the elasticsearch group ids that are listed under the language code supplied.
    :param es_client: (elasticsearch.Elasticsearch) The connection object to elasticsearch
    :param language_code: (str) The language of groups to filter with
    :param group_index: (str) The elasticsearch index that stores SoCaTel co-creation groups. By default, "so_group"
    :return: (list) List of SoCaTel group ids
    """
    q_body = {
        "_source": ["group_id"],
        "query": {
            "match": {
                "language.language_code": language_code
            }
        }
    }

    resp = es_client.search(
        index=group_index,
        body=q_body
    )

    return [x['_source']['group_id'] for x in resp['hits']['hits']]


def get_group_lang(es_client, group_id, group_index='so_group'):
    """
    Retrieve the language code supplied.
    :param es_client: (elasticsearch.Elasticsearch) The connection object to elasticsearch
    :param group_id: (int) The id of the elasticsearch index storing SoCaTel co-creation groups.
    :param group_index: (str) The elasticsearch index that stores SoCaTel co-creation groups. By default, "so_group"
    :return: (str or None) If the group is found, its language code is returned, else None.
    """
    q_body = {
        "_source": [
            "language.language_code",
            "group_id"
        ],
        "query": {
            "match": {
                "group_id": str(group_id)
            }
        }
    }

    resp = es_client.search(
        index=group_index,
        body=q_body,
    )

    if len(resp['hits']['hits']) == 1:
        return resp['hits']['hits'][0]['_source']['language']['language_code']
    else:
        return None


def load_translation_model(language_code):
    """
    Given a language code, the corresponding pre-trained tokenization and transformer translation model are returned
    for the pair <lang_code> to English.
    :param language_code: (str) The target language code
    :return: (transformers.AutoTokenizer, transformers.AutoModelForSeq2SeqLM) If the language code is not supported for
    translation to English else (None, None)
    """
    tokenizer = None
    model = None

    if language_code != "en":
        try:
            tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(language_code))
            model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-{}-en".format(language_code))
        except Exception:
            print('Language model for {} cannot be found'.format(language_code))

    return tokenizer, model


def translate_text_to_en(text, tokenizer, model):
    """
    Using the pre-trained tokenization and transformer translation model retrieved from the `load_translation_model`
    function, invoke model inference using the supplied, non-english text to return its English translation.
    :param text: (str) The text to be translated
    :param tokenizer: (transformers.AutoTokenizer) Transformer model tokenizer
    :param model: (transformers.AutoModelForSeq2SeqLM) Transformer sequence to sequence model
    :return: (str) The translated text
    """
    if tokenizer is None:
        return text

    batch = tokenizer.prepare_seq2seq_batch(src_texts=[text])  # don't need tgt_text for inference
    gen = model.generate(**batch)  # for forward pass: model(**batch)
    translated_text = tokenizer.batch_decode(gen, skip_special_tokens=True)

    return translated_text


def calculate_current_sentiment_score(es_client, update_from, update_to, language_code="en", posts_index='so_post'):
    """
    Based on a give time period, between `update_from` and `update_to`, search for SoCaTel group posts and identify
    sentiment by scoring it with the PIO docker instance (utilising linear regression at the moment).
    :param es_client: The connection object to elasticsearch
    :param update_from: Unix timestamp specifying the point from which to make updates.
    :param update_to: Unix timestamp specifying the point until updates are to be made (current execution time).
    :param language_code: The language of groups to filter with.
    :param posts_index: The elasticsearch index hosting user posts.
    :return: (dict) Group ids mapped to their group average sentiment over a 1-day window (if this is not a fresh run).
    """
    tokenizer, model = load_translation_model(language_code=language_code)
    group_ids = get_groups_by_lang(es_client, language_code)

    q_body = {
        "size": 100,
        "query": {
            "bool": {
                "filter": {
                    "terms": {
                        "group_id": group_ids
                    }
                },
                "must": [
                    {
                        "range": {
                            "post_timestamp": {
                                "gte": update_from,
                                "lt": update_to
                            }
                        }
                    }
                ]
            }
        }
    }

    # make a search() request to get all docs in the index
    resp = es_client.search(
        index=posts_index,
        body=q_body,
        scroll='10m'  # length of time to keep search context
    )

    del q_body

    # keep track of past scroll _id
    old_scroll_id = resp['_scroll_id']

    print("\nParsing through posts for sentiment extraction.\n")
    new_group_scores = {}

    progress_bar = tqdm(total=len(resp['hits']['hits']))
    while len(resp['hits']['hits']):
        # iterate over the document hits for each 'scroll'
        for source in resp['hits']['hits']:
            try:
                # Create a new user vector
                doc = source['_source']

                # Send this to sentiment API
                post_text = " ".join(pre_process_free_text(doc['post_text'], '1', False, []))
                if len(post_text) == 0:
                    progress_bar.update(1)
                    continue

                # English could be posted in SoCaTel groups from non-English speaking localities. Identification of
                # language could have been an extra addition but we rely on locality for now.
                post_text = post_text if language_code == "en" else translate_text_to_en(post_text, tokenizer, model)[0]
                # Try to ensure that 'latin-1' encoding won't fail when passed to the sentiment classifier.
                post_text = ''.join(i for i in post_text if ord(i) < 128)

                try:
                    sentiment = requests.post(cfg['sentiment_api']['url'], data=str({"text": post_text})).json()
                except UnicodeEncodeError:
                    progress_bar.update(1)
                    continue

                if sentiment['confidence'] > 0.85:  #
                    if str(doc['group_id']) not in new_group_scores:
                        new_group_scores[str(doc['group_id'])] = {
                            'values': [float(sentiment['category'])],
                            'total_posts': 1
                        }
                    else:
                        new_group_scores[str(doc['group_id'])]['values'].append(float(sentiment['category']))
                        new_group_scores[str(doc['group_id'])]['total_posts'] += 1
                progress_bar.update(1)
            except KeyError as ex:
                # We've fallen here if there were no posts..
                print(ex)
                progress_bar.update(1)
                continue

        # Make a new request using the Scroll API
        resp = es_client.scroll(
            scroll_id=old_scroll_id,
            scroll='10m'
        )

        # check if there's a new scroll ID
        if old_scroll_id != resp['_scroll_id']:
            print("NEW SCROLL ID:", resp['_scroll_id'])

        # keep track of past scroll _id
        old_scroll_id = resp['_scroll_id']

    progress_bar.close()
    return new_group_scores


def update_old_sentiment_score(database, group_scores, time_from, time_to, algo_name):
    """
    Based on the number of new averages, over the windows of a day if this is not a fresh restart, re-calculate group
    average sentiment in MongoDB.
    :param database: Connection object to MongoDB
    :param group_scores: (dict) Holds group ids and their group average sentiment over a 1-day window (if this is not
    a fresh run).
    :param time_from: Unix timestamp specifying the point from which to make updates.
    :param time_to: Unix timestamp specifying the point until updates are to be made (current execution time).
    :param algo_name: The name of the algorithm that was used to produce these results.
    :return: None
    """
    date_from = datetime.fromtimestamp(time_from / 1e3)
    date_to = datetime.fromtimestamp(time_to / 1e3)

    for group_id in group_scores.keys():
        avg_score = np.array(group_scores[group_id]['values']).mean()
        total_posts = group_scores[group_id]['total_posts']

        entry = {
            "date_from": date_from,
            "date_to": date_to,
            "group_id": group_id,
            "avg_sentiment_score": avg_score,
            "no_of_posts": total_posts
        }

        database.general[algo_name].insert_one(entry)


if __name__ == "__main__":
    print("Scoring group posts based on their general sentiment")

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo_name', dest='algo_name',
                        help="Which algorithm should we use to generate sentiment scores? The database to save results "
                             "also depends on this.",
                        type=str, required=False, default='linear_regression')
    parser.add_argument('--lang_codes', nargs="+",
                        dest='lang_codes', help='Filter groups to be scored based on their language. '
                                                              'Select from  either combination of '
                                                              '["en", "es", "fi", "hu"]',
                        required=False, default=["en", "es", "fi", "hu"])
    args = parser.parse_args()

    cfg_f_path = "config.yml"
    if __package__ is None:
        cfg_f_path = "../../" + cfg_f_path

    with open(cfg_f_path, 'r') as uim_yaml_file:
        cfg = yaml.load(uim_yaml_file, Loader=yaml.FullLoader)

    # 1. Get last run time
    init_job = check_if_init_job(cfg, 'so_group', 'sentiment')
    until_run_ts = int(datetime.today().timestamp()) * 1000

    last_run_ts = 0
    # Assume a lag of 1 day
    if not init_job:
        last_run_ts = int(datetime.timestamp(datetime.today() - timedelta(days=1)) * 1000)

    # 2. Make API calls to the PIO sentiment docker for posts between the last run time and now
    es = Elasticsearch(['http://' + cfg['elasticsearch']['user'] + ':' +
                        urlquote(cfg['elasticsearch']['passwd']) + '@' +
                        cfg['elasticsearch']['host'] + ':' +
                        cfg['elasticsearch']['port']],
                       verify_certs=True)

    for lang_code in args.lang_codes:
        grp_scores = calculate_current_sentiment_score(es, last_run_ts, until_run_ts, lang_code)

        print("Done for language with code {}.. Inserting into MongoDB".format(lang_code))
        # 3. Calculate the average of the two time windows for each group. If new groups were found, insert them.
        db, client = get_mongo_resources(cfg, 'so_group', 'sentiment')

        update_old_sentiment_score(db, grp_scores, last_run_ts, until_run_ts, args.algo_name)

    print("Finished.. Exiting")
    client.close()
