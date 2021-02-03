import os
import yaml
import traceback

from src.scorers.utils import *
from src.scorers.uim_scorer import UserToItemScorer
from urllib.parse import quote_plus as urlquote
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--item_index', dest='item_index', help="Which elasticsearch index characterises items?",
                            type=str, required=False, default='so_group')
        parser.add_argument('--iim_mtx_path', dest='iim_mtx_path', help="Path to `item to item matrix` (CSV only).",
                            type=str, required=False, default=None)
        parser.add_argument('--uim_mtx_path', dest='uim_mtx_path', help="Path to `user to item matrix` (CSV only).",
                            type=str, required=False, default=None)
        parser.add_argument('--last_run_ts', dest='last_run_ts',
                            help="Unix timestamp. In case you want to manually set the "
                                 "time from which data will be retrieved.",
                            type=str, required=False, default=None)
        parser.add_argument('--until_run_ts', dest='until_run_ts',
                            help="Unix timestamp. In case you want to manually set the "
                                 "time until which data will be retrieved.",
                            type=str, required=False, default=None)

        args = parser.parse_args()

        print("---")
        print(str(datetime.now()))
        print("User to " + str(args.item_index) + " recommendation handler running...")

        iim_mtx = None
        if args.iim_mtx_path is not None:
            if os.path.exists(args.iim_mtx_path):
                iim_mtx = pd.read_csv(args.iim_mtx_path)
            else:
                print("Could not find item to item matrix path. Recreating..")

        uim_mtx = None
        if args.uim_mtx_path is not None:
            if os.path.exists(args.uim_mtx_path):
                uim_mtx = pd.read_csv(args.uim_mtx_path)
            else:
                print("Could not find user to item matrix path. Recreating..")

        if args.until_run_ts is None:
            until_ts = int(datetime.today().timestamp() * 1000)
        else:
            until_ts = args.until_run_ts

        if args.last_run_ts is None:  # Take the last day
            last_run = datetime.today() - timedelta(days=1)
            last_run_ts = int(last_run.timestamp() * 1000)
        else:
            last_run_ts = args.last_run_ts

        with open("config.yml", 'r') as uim_yaml_file:
            cfg = yaml.load(uim_yaml_file, Loader=yaml.FullLoader)

        uim_yaml_file.close()

        # SoCaTel machine
        es = Elasticsearch(['http://' + cfg['elasticsearch']['user'] + ':' +
                            urlquote(cfg['elasticsearch']['passwd']) + '@' +
                            cfg['elasticsearch']['host'] + ':' +
                            cfg['elasticsearch']['port']],
                           verify_certs=True)

        init_job = check_if_init_job(cfg, args.item_index)

        uim_scorer = UserToItemScorer(
            client=es,
            history_index=cfg['elasticsearch']['history_index'],
            item_index=args.item_index,
            min_score=cfg['scoring']['min_score'],
            max_score=cfg['scoring']['max_score'],
            iim_mtx=iim_mtx,
            uim_mtx=uim_mtx,  # Old snapshot of the user to item matrix?
            update_from=last_run_ts,
            update_to=until_ts,
            cfg=cfg,
            is_init=init_job
        )

        # =============
        # For each action in the history track, re-evaluate user score per service
        # =============
        local_uim_before = None
        if init_job:
            total_hits_processed = uim_scorer.update_from_all_history_posts()
            local_uim = uim_scorer.uim_mtx
        else:
            if uim_mtx is None:
                # Locate the user-to-item matrix and read it in as a pandas.DataFrame()
                local_uim_before = get_score_mtx_from_mongo(cfg, args.item_index, is_iim=False)
                uim_scorer.uim_mtx = local_uim_before
            else:
                uim_scorer.uim_mtx = uim_mtx
                del uim_mtx

            if iim_mtx is None:
                # Only allow same item similarity in this handler
                local_iim_before = get_score_mtx_from_mongo(cfg, args.item_index, is_iim=True,
                                                            secondary_idx=args.item_index)

                # local_iim_before.index = [int(x) for x in local_iim_before.index]
                # local_iim_before_sorted = local_iim_before.sort_index()
                local_iim_before_sorted, needs_expansion = expand_iim_if_necessary(es, local_iim_before,
                                                                                   args.item_index, None, cfg)
                uim_scorer.iim_mtx = local_iim_before_sorted
                del local_iim_before
                del local_iim_before_sorted
            else:
                uim_scorer.iim_mtx = iim_mtx

            total_hits_processed = uim_scorer.update_from_latest_history_posts()
            local_uim = uim_scorer.uim_mtx  # Retrieve the updated user to item matrix

        local_uim_pio, local_uim_pio_non_zeroes, local_uim_pio_non_zeroes_str = pio_uim_transform(
            local_uim, as_str=True, is_iim=False)  # Transform to pandas

        del local_uim
        batch_mongo_insert_pio_data(local_uim_pio, cfg, args.item_index)  # Used by the pio engine.
        batch_mongo_insert_pio_data(local_uim_pio_non_zeroes, cfg, args.item_index, non_zeroes=True)

        # Save to txt if there were any updates or if this is the first time running this script.
        if local_uim_before is None or local_uim_before.equals(local_uim_pio):
            with (open(cfg['local'][args.item_index]['path_to_user_item_matrix'], 'w')) as f:
                f.write(local_uim_pio_non_zeroes_str)

            f.close()

        # =============
        # Print some stats
        # =============
        print("Total hits processed: {}".format(str(total_hits_processed)))
        print("---")
    except Exception as ex:
        traceback.print_exc()
        print("---")
