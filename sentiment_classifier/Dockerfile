FROM python:3.6-slim-stretch

# Install cron
RUN apt-get update && \
    apt-get -y install cron nano

# Copy files from local repo. Assumes they are stored in root dir
WORKDIR /socatel/reco/
COPY ../requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html && \
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
