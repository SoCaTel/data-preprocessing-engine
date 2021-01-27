import os
import sys
import json
import datetime
import requests
import subprocess
import pandas as pd
import numpy as np

module_path = os.path.abspath(os.path.join('../src/scorers'))

if module_path not in sys.path:
    sys.path.append(module_path)

import utils


def process_text(entry, f_json):
    """
    Use in conjunction with pandas apply function. Preprocess the (tweet) text and clean it from:
    1. Pre-selected keywords
    2. Stopwords
    3. Special characters
    4. Character repetition
    5. Numeric characters
    :param entry: (pandas.Series) A row from a pandas dataframe containing at least the column 'tweet_text' and
    'num_sentiment', the latter corresponding to the sentiment score from a scale of 0 - 1 (<0.5 negative,
    >0.5 positive)
    :param f_json: (file handler) Used to write output to file for future reference of the sentiment classification
    engine.
    :return: (str) The cleaned text
    """
    global entity_counter
    global start_time

    if entry['tweet_text'] is not np.nan:
        try:
            clean_text_l = utils.pre_process_free_text(entry['tweet_text'], lang_id=1, rm_stopwords=False)

            if len(clean_text_l) >= MINIMUM_WORD_COUNT:
                clean_text = ' '.join(clean_text_l)
                clean_text = clean_text.replace("'", "")
                start_time += datetime.timedelta(minutes=1)
                start_t_str = str(start_time).split(" ")

                json_struct = {
                    "eventTime": "T".join(start_t_str),
                    "entityId": entity_counter,
                    "entityType": "source",
                    "properties": {
                        "phrase": clean_text,
                        "sentiment": entry['num_sentiment']
                    },
                    "event": "phrases"
                }

                entity_counter += 1

                f_json.write(json.dumps(json_struct) + "\n")
                return clean_text
            else:
                return None
        except Exception as ex:
            print(ex)
            return None


def prepare_properties_column(row):
    """
    Returns the format which predictionio needs to process and train a sentiment classification engine
    :param row: (pandas.Series) A row from a pandas dataframe containing at least the column 'clean_tweet_text' and
    'num_sentiment', the latter corresponding to the sentiment score from a scale of 0 - 1 (<0.5 negative,
    >0.5 positive)
    :return: (dict) The line output in the form of a dictionary
    """
    return {"phrase": row['clean_tweet_text'], "sentiment": row['num_sentiment']}


"""
--- Sentiment towards products' file conversion
This file has been selected as group conversations are usually geared towards judging existing services and commenting
about newly proposed services. Give this, it is clear that this resembles sentiment towards products/brands, which the
following dataset captures.

To complement this dataset, we also use the movie sentiment dataset already available, but only use the very 
positive/negative comments made to remove noise from more neutral comments.
"""


if __name__ == "__main__":
    MINIMUM_WORD_COUNT = 3

    start_time = pd.to_datetime("2015-06-08T16:58:14.285+0000")

    if not os.path.exists('judge-1377884607_tweet_product_company.csv'):
        r = requests.get("https://query.data.world/s/5gxidpupmkcesf43vkltv4h6s7erlv")
        open('judge-1377884607_tweet_product_company.csv', 'wb').write(r.content)

    sent_dataset_1 = pd.read_csv("judge-1377884607_tweet_product_company.csv", encoding='latin-1')
    sent_dataset_1.is_there_an_emotion_directed_at_a_brand_or_product.value_counts()

    # Filter for rows of which sentiment was recognised
    sent_dataset_1_f = sent_dataset_1[sent_dataset_1['is_there_an_emotion_directed_at_a_brand_or_product'].isin([
        'Positive emotion', 'Negative emotion'
    ])].copy()

    sent_dataset_1_f['num_sentiment'] = 0
    sent_dataset_1_f.loc[sent_dataset_1_f[
                             'is_there_an_emotion_directed_at_a_brand_or_product'] == 'Positive emotion',
                         'num_sentiment'] = 1

    sent_dataset_1_f['num_sentiment'].value_counts()
    entity_counter = 0

    f = open('product_sentiment.json', 'w')
    sent_dataset_1_f['clean_tweet_text'] = sent_dataset_1_f.apply(process_text, axis=1, f_json=f)
    f.close()

    # Prepare a JSON file in the format that PIO will accept it.
    sent_dataset_1_f['entityId'] = sent_dataset_1_f.index
    sent_dataset_1_f['entityType'] = "source"
    sent_dataset_1_f['properties'] = sent_dataset_1_f.apply(prepare_properties_column, axis=1)
    sent_dataset_1_f['event'] = "phrases"

    sent_dataset_1_json = sent_dataset_1_f[['entityId', 'entityType', 'properties', 'event']]
    sent_dataset_1_f.to_json("final_sentiment.json")

    # Convert the default sentiment file provided by predictionio to 0 - 1 scale
    default_f = open('sentimentanalysis.json', 'r')
    new_default = open('clean_default_sentiment_analysis.json', 'w')

    sentiment_map = {0: 0, 4: 1}
    unique_words = set()

    for line in default_f.readlines():
        line_json = json.loads(line)
        sent = line_json['properties']['sentiment']
        if sent in list(sentiment_map.keys()):
            new_sent = sentiment_map[sent]  # Re-map to a 0 - 1 scale.
            line_json['properties']['sentiment'] = new_sent

            line_json['properties']['phrase'] = line_json['properties']['phrase'].replace("'", "")  # .lower()
            words = line_json['properties']['phrase'].split(" ")
            unique_words.update(words)
            if len(words) >= MINIMUM_WORD_COUNT:
                start_time += datetime.timedelta(minutes=1)
                start_time_str = str(start_time).split(" ")

                # re-organise entityIds
                line_json['eventTime'] = "T".join(start_time_str)

                line_json['entityId'] = entity_counter  # Continue from where we left off.
                entity_counter += 1

                new_default.write(json.dumps(line_json) + '\n')

    default_f.close()
    new_default.close()

    len(unique_words)

    try:
        output = subprocess.check_output(
            ['bash', '-c',
             "cat 'product_sentiment.json' >> 'clean_default_sentiment_analysis.json'"])
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    print("Dataset created under ./ 'clean_default_sentiment_analysis.json' w/ {} rows".format(entity_counter))
