import os
import yaml
import traceback

from src.scorers.utils import *
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from urllib.parse import quote_plus as urlquote
from src.scorers.iim_scorer import ItemToItemScorer

if __name__ == "__main__":
    try:
        """
        Use this handler to create/update item to item matrices without a dependency on the creation/update on a user to 
        item matrix. 
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--item_index', dest='item_index', help="Which elasticsearch index characterises items?",
                            type=str, required=False, default='so_group')
        parser.add_argument('--sec_item_index', dest='sec_item_index', help="Secondary elasticsearch index to produce "
                                                                            "similarities with items from `item_index`?",
                            type=str, required=False, default=None)
        parser.add_argument('--iim_mtx_path', dest='iim_mtx_path', help="Path to `item to item matrix` (CSV only).",
                            type=str, required=False, default=None)
        parser.add_argument('--last_run_ts', dest='last_run_ts',
                            help="Unix timestamp. In case you want to manually set the time from which data will be "
                                 "retrieved.",
                            type=str, required=False, default=None)
        parser.add_argument('--until_run_ts', dest='until_run_ts',
                            help="Unix timestamp. In case you want to manually set the "
                                 "time until which data will be retrieved.",
                            type=str, required=False, default=None)

        args = parser.parse_args()

        print("---")
        print(str(datetime.now()))
        if args.sec_item_index is None:
            print(str(args.item_index) + " to " + str(args.item_index) + " recommendation handler running...")
        else:
            print(str(args.item_index) + " to " + str(args.sec_item_index) + " recommendation handler running...")

        with open("config.yml", 'r') as uim_yaml_file:
            cfg = yaml.load(uim_yaml_file, Loader=yaml.FullLoader)

        uim_yaml_file.close()

        if args.sec_item_index is None:
            init_job = check_if_init_job(cfg, args.item_index, args.item_index)
        else:
            init_job = check_if_init_job(cfg, args.item_index, args.sec_item_index)

        # --------------------------------------------
        # Find and retrieve the item to item matrix  -
        # --------------------------------------------
        if not init_job:
            iim_mtx = get_score_mtx_from_mongo(cfg, args.item_index, is_iim=True, secondary_idx=args.sec_item_index)
        elif args.iim_mtx_path is not None:
            if os.path.exists(args.iim_mtx_path):
                iim_mtx = pd.read_csv(args.iim_mtx_path)
            else:
                print("Could not find item to item matrix path. Recreating..")
        else:
            iim_mtx = None

        if args.until_run_ts is None:
            until_ts = int(datetime.today().timestamp() * 1000)
        else:
            until_ts = args.until_run_ts

        if args.last_run_ts is None:  # Take the last day
            last_run = datetime.today() - timedelta(days=1)
            last_run_ts = int(last_run.timestamp() * 1000)
        else:
            last_run_ts = args.last_run_ts

        # SoCaTel machine
        es = Elasticsearch(['http://' + cfg['elasticsearch']['user'] + ':' +
                            urlquote(cfg['elasticsearch']['passwd']) + '@' +
                            cfg['elasticsearch']['host'] + ':' +
                            cfg['elasticsearch']['port']],
                           verify_certs=True)

        # If item to item matrix was not found
        if iim_mtx is None:
            # Produce it.
            iim_scorer = ItemToItemScorer(
                client=es,
                item_index=args.item_index,
                sec_item_index=args.sec_item_index,
                update_from=last_run_ts,
                update_to=until_ts,
                cfg=cfg
            )

            iim = iim_scorer.get_item_to_item_matrix()
            iim_pio, iim_pio_non_zeroes, _ = pio_uim_transform(iim, is_iim=True)

            if args.sec_item_index is None:
                batch_mongo_insert_pio_data(iim_pio, cfg, args.item_index, args.item_index)
                batch_mongo_insert_pio_data(iim_pio_non_zeroes, cfg, args.item_index, args.item_index, non_zeroes=True)
            else:
                batch_mongo_insert_pio_data(iim_pio, cfg, args.item_index, args.sec_item_index)
                batch_mongo_insert_pio_data(iim_pio_non_zeroes, cfg, args.item_index, args.sec_item_index, non_zeroes=True)
        # Check if the one found/given is up to date
        else:
            iim_mtx, needs_expansion = expand_iim_if_necessary(es, iim_mtx, args.item_index, args.sec_item_index, cfg)

            if needs_expansion:
                # Insert back to MongoDB, overwriting the older records
                iim_pio, iim_pio_non_zeroes, _ = pio_uim_transform(iim_mtx, is_iim=True)
                if args.sec_item_index is None:
                    batch_mongo_insert_pio_data(iim_pio, cfg, args.item_index, args.item_index)
                    batch_mongo_insert_pio_data(iim_pio_non_zeroes, cfg, args.item_index, args.item_index, non_zeroes=True)
                else:
                    batch_mongo_insert_pio_data(iim_pio, cfg, args.item_index, args.sec_item_index)
                    batch_mongo_insert_pio_data(iim_pio_non_zeroes, cfg, args.item_index, args.sec_item_index,
                                                non_zeroes=True)

        print("Exiting...")
        print("---")
    except Exception as ex:
        traceback.print_exc()
        print("---")
