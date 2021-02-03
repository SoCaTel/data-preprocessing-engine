import yaml

from tqdm import tqdm
from elasticsearch import Elasticsearch
from datetime import datetime
from src.scorers.utils import *
from urllib.parse import quote_plus as urlquote


if __name__ == "__main__":
    print("---")
    print(str(datetime.now()))
    print("User to services recommendations based on user activity in groups and similarity between groups and services")

    with open("config.yml", 'r') as uim_yaml_file:
        cfg = yaml.load(uim_yaml_file, Loader=yaml.FullLoader)

    uim_yaml_file.close()

    # -- Some boilerplate

    gts_db, client = get_mongo_resources(cfg, cfg['elasticsearch']['groups_index'],
                                         cfg['elasticsearch']['services_index'])

    gts_coll_name = cfg['elasticsearch']['groups_index'] + '_to_' + cfg['elasticsearch']['services_index']

    uts_db = client[cfg['mongo']['user'][cfg['elasticsearch']['services_index']]['db_name']]
    uts_coll_name = 'user_to_' + cfg['elasticsearch']['services_index']

    client.close()

    # Re-do every time. It is hard to update this, as it has 2 dependencies. The user activity changes in a group, and
    # similarity scores between existing groups and new services are hard to track. The fact that user activity is
    # aggregated, makes this computationally feasible.
    if uts_coll_name in uts_db.collection_names():
        uts_db[uts_coll_name].drop()

    # -- End of boilerplate

    # SoCaTel machine
    es = Elasticsearch(['http://' + cfg['elasticsearch']['user'] + ':' +
                        urlquote(cfg['elasticsearch']['passwd']) + '@' +
                        cfg['elasticsearch']['host'] + ':' +
                        cfg['elasticsearch']['port']],
                       verify_certs=True)

    # For all users, calculate their user activity and update their score.
    match_all = {
        "size": 100,
        "_source": ["user_id"],
        "query": {
            "match_all": {}
        }
    }

    # make a search() request to get all docs in the index
    resp = es.search(
        index=cfg['elasticsearch']['users_index'],
        body=match_all,
        scroll='10m'  # length of time to keep search context
    )

    del match_all

    # keep track of past scroll _id
    old_scroll_id = resp['_scroll_id']

    print("Parsing through all users")
    progress_bar = tqdm(total=len(resp['hits']['hits']))
    while len(resp['hits']['hits']):
        # iterate over the document hits for each 'scroll'
        for source in resp['hits']['hits']:
            # Create a new user vector
            doc = source['_source']
            user_id = doc['user_id']

            get_user_activity = {
                "size": 0,
                "query": {
                    "bool": {
                        "filter": {
                            "terms": {
                                "user_id": [user_id]
                            }
                        }
                    }
                },
                "aggs": {
                    "groups": {
                        "terms": {
                            "field": "group_id",
                            "size": 5
                        }
                    }
                }
            }

            resp = es.search(
                index=cfg['elasticsearch']['history_index'],
                body=get_user_activity
            )

            for bucket in resp['aggregations']['groups']['buckets']:
                group_id = str(bucket['key'])
                total_user_actions = bucket['doc_count']

                collection = gts_db[gts_coll_name]
                regex_find = re.compile("^" + str(group_id) + '::.*')
                cursor = collection.find({'combo': regex_find})

                for document in cursor:
                    score = float(document['combo'].split("::")[-1])

                    if score > 0:
                        uts_score = total_user_actions * score
                        service_id = int(document['combo'].split("::")[1])

                        try:
                            # insert info to Mongo.
                            uts_db[uts_coll_name].insert(
                                {'combo': user_id + '::' + str(service_id) + '::' + str(uts_score)}
                            )
                        except WriteError as we:
                            print(we)

            progress_bar.update(1)

        # Make a new request using the Scroll API
        resp = es.scroll(
            scroll_id=old_scroll_id,
            scroll='10m'
        )

        # check if there's a new scroll ID
        if old_scroll_id != resp['_scroll_id']:
            print("NEW SCROLL ID:", resp['_scroll_id'])

        # keep track of past scroll _id
        old_scroll_id = resp['_scroll_id']

    progress_bar.close()
