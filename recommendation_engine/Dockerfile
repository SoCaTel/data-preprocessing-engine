FROM python:3.6-slim-stretch

MAINTAINER CyRIC "info@cyric.eu"

# Install cron
RUN apt-get update && \
    apt-get -y install cron nano

# Copy files from local repo. Assumes they are stored in root dir
WORKDIR /socatel/reco/
COPY ../requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /var/log/ && \
    touch /var/log/cron.log

COPY ../config.yml ./config.yml
COPY ../src/scorers/* ./src/scorers/

# Download stopwords
RUN python3 -W ignore -m nltk.downloader stopwords

ADD cronpy /socatel/reco/cronpy
RUN cat /socatel/reco/cronpy >> /etc/crontab && \
    crontab /etc/crontab

ENTRYPOINT ["/usr/sbin/cron", "-f"]
