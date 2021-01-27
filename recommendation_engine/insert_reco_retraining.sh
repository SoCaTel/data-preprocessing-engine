docker exec -t docker_pio_1 touch /var/log/cron.log
docker cp ./import_new_events.sh docker_pio_1:/templates/import_new_events.sh
docker exec -t docker_pio_1 chmod 755 /templates/import_new_events.sh
