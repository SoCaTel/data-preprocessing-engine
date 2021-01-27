# Recommendation engine data processing module

This project contains tools and processes to produce and store content-based recommendations between SoCaTel entities 
(groups, organisations, services, etc.) and also aggregate and store user activity within groups. The latter is used as 
an input to a PredictionIO module, which will produce collaborative-based recommendations of SoCaTel entities back to 
users. These are computed daily, so that the algorithm can pick up updates on the platform and modify, if necessary, 
said recommendations.

## Getting Started
### Prerequisites

All tools and processes are meant to be deployed under a docker container. To get everything set up, you should:

* Have docker installed in your machine. See [this](https://runnable.com/docker/install-docker-on-linux) for a general 
guide on Linux machines (preferred OS for this project).
* Have already deployed a MongoDB docker instance within a docker network. See instructions in the next section on how 
to set this up.
* GraphQL, for semantic repository queries, exposed through API calls in [this](https://github.com/SoCaTel/rest-api) 
repository.
* Elasticsearch, for direct queries to the database. For help in setting up the Elasticsearch indices, refer to [this
](https://github.com/SoCaTel/elasticsearch-schema) repository.
* After deploying a docker container, take a note of the docker instance's id, referenced as `<mongo_image_id>` here.
* Have a (virtual) python3 environment up and running with `predictionio` installed. Do so with either:
```
pip install predictionio
# or
easy_install predictionio
```
* For optimal performance, the data handlers need to be deployed and have already gathered tweets from organisations' 
twitter accounts.

**NOTE** This repository has been developed using Python 3.6

### Database setup
The project relies on a MongoDB for reliable storage of recommendations. To set one up, you need to:

* Pull a fresh container from [here](https://hub.docker.com/_/mongo).
* Switch to the cloned project's directory, `cd ~/engine_data_preprocessing`.
* Edit the `stack.yml` file with the port `<mongo_port>`, the username `<mongo_username>` and password 
`<mongo_password>` of your choice.
* Run `docker-compose -f stack.yml up --detach` to initialise a container hosting an instance of MongoDB in the background.

## Configurations

Under the [configuration](../config.yml) file in the root folder, change the following based on your current setup:

* `mongo.host`: Use the `<mongo_image_id>`.
* `mongo.port`: Change this to `<mongo_port>`, where your MongoDB host exposes the database on the docker network setup.
* `mongo.username`: Change to one of your choosing.
* `mongo.password`: Change to one of your choosing.

The rest of the configurations concern the schema of the elasticsearch index hosting the SoCaTel raw data and the 
GraphQL endpoints, which have been pre-discussed and should not change. To create the indices with their
corresponding schema, refer to the `elasticsearch-schema` repository of the SoCaTel group. If you are deploying on a
new ecosystem, you will have to at least change the following in the [configuration](../config.yml) file:

* `elasitcsearch.host`
* `elasitcsearch.port`
* `elasitcsearch.user`
* `elasitcsearch.passwd`
* `graphql.graphQL_endpoint`
* `graphql.graphQL_authorisation`

## Deployment
### Pre-processing container

Note that the following will be run in the sub-folder where this README file resides. The Dockerfile will reference the
necessary files/folders from the root directory.

1. The name of the network that the MongoDB docker container runs under should be `recommendation-engine-net`. Verify 
this by running:
`docker inspect <mongo_image_id> | grep -A 2 "Networks"`

    The second line contains the network's name, referred to as `mongodb_network_name` here. Keep a note of this for 
    later. 

2. In the directory of this README file run:
`docker build --no-cache -t <recommendation_docker_id>:<tag> .`

    This will install all python libraries in the `../requirements.txt` file and copy the files necessary for executing 
    the daily cron jobs inside the container.

4. Following this, initialise a new container with:
`docker run -dit --name <recommendation_docker_id> --restart always -v ~/recommendationengine/docker_out:/socatel/reco/docker_out --network <mongodb_network_name> <recommendation_docker_id>:<tag>`

The `docker_out` folder is mapped to the container's equivalent folder, for the PIO docker container to pick up text 
files used as input to the model training procedures. See the next section to set up and deploy the PIO docker based on 
the output of this.

### Project structure
The final folder structure, within the docker container, should look like:

```
.
├── config.yml
├── cronpy
├── requirements.txt
└── src
    └── scorers
        ├── iim_scorer.py
        ├── item_similarity_handler.py
        ├── sentiment_scorer.py
        ├── uim_scorer.py
        ├── user_to_item_rec_handler.py
        ├── user_to_services_rec_handler.py
        └── utils.py
```

### PIO engine container

1. Get the latest version of PredictionIO from SoCaTel's git repository using: 
`git clone https://github.com/SoCaTel/predictionio.git`
2. Switch to and build the image using:
    ```
    cd predictionio/docker
    docker build --no-cache -t predictionio/pio pio
    ```

3. PredictionIO requires a properly configured storage to work in its full potential. We will use PostgreSQL. Build and 
run the containers (in the background) using the preset compose files, while still in the `predictionio/docker` 
directory, :
```
docker-compose -f docker-compose.yml \
    -f pgsql/docker-compose.base.yml \
    -f pgsql/docker-compose.meta.yml \
    -f pgsql/docker-compose.event.yml \
    -f pgsql/docker-compose.model.yml \
    up --detach
```

4. Set the pio-docker command tool to the default execution path using: ```export PATH=`pwd`/bin:$PATH```
5. Enter templates directory (`predictionio/docker/templates`) and download the SoCaTel recommendation engine template 
using: 
`git clone https://github.com/SoCaTel/predictionio-template-recommender.git RecoEngine`
6. In `predictionio/docker/templates/RecoEngine/` directory, run `pio-docker app new socatel_reco` to create a 
new engine. When the engine is created, an "Access Key" will be given, referred to as `<your_access_key>` here. 
Make a note of it.
7. Change the `datasource.params.appName` to `socatel_reco` under 
`./predictionio/docker/templates/RecoEngine/engine.json`
8. Replace all placeholders in `import_new_events.sh` as described in previous steps.
9. Run `./insert_reco_retraining.sh` to copy over the file to your PredictionIO docker repository (by default this is
`docker_pio_1`). Change accordingly.
10. Execute `touch /var/log/cron.log` on your host machine.
11. Finally, add the following to your `/etc/crontab` on the host machine to schedule daily insertions and engine 
retraining.
```
0 7    * * *   cd /path/to/predictionio/docker && export PATH=`pwd`/bin:$PATH && docker exec -i docker_pio_1 /bin/bash /templates/import_new_events.sh >> /var/log/cron.log 2>&1
```
12. Run `crontab /etc/crontab` to enable the above.

## Built With

* [Docker](https://www.docker.com/) - The secure way to build and share this application, anywhere.
* [MongoDB](https://www.mongodb.com/) - A general purpose distributed data store, geared towards document-based 
instances.
* [Pandas](https://pandas.pydata.org/) - Used for data import, transformations and data wrangling.
* [PredictionIO](https://predictionio.apache.org/) - Configured our output data format to match that of templates hosted
 using this service.

## **Contact**
If you encounter any problems, please contact the following:

[<img src="https://www.cyric.eu/wp-content/uploads/2017/04/cyric_logo_2017.svg" alt="CyRIC | Cyprus Research and Innovation Centre" width="150" />](mailto:info@cyric.eu)

## License

[Apache-2.0](../LICENSE)
