#!/bin/bash

cd /templates/ || exit

MODIFIED=$(find /soca_reco_out/local_uim_group_test.txt -mtime -1 -type f -print)

# If variable is non empty - meaning that the file has been modified, re-deploy the PIO engine.
if [ ! -z "$MODIFIED" ]; then
  # Delete data first, as we want to override with new group "ratings"
  pio app data-delete socatel_reco --force
  python RecoEngine/data/import_eventserver.py --access_key <your_access_key> --file /soca_reco_out/local_uim_group_test.txt
  cd RecoEngine || return
  SUCCESS=1
  ATTEMPTS=1

  pio build
  while [ $SUCCESS -eq 1 ] && [ $ATTEMPTS -le 5 ]
  do
    ATTEMPTS=$(( ATTEMPTS + 1 ))
    pio train # -- Could fail. Try 5 times
    SUCCESS=$?
  done
  dt=$(date '+%d/%m/%Y %H:%M:%S');
  echo "Re-deployed recommendation engine with new events at $dt"
  pio deploy
fi

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Finished re-checking for new events to import at $dt"
