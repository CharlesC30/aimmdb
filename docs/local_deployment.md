# AIMMDB Local Deployment

This document describes setting up a local deployment of aimmdb suitable for development.

## Installation

First we create a fresh conda environment and install aimmdb.
This will also install all dependencies including tiled.
Since we are setting up a local deployment for development we install aimmdb in editable mode (`pip install -e`) so that changes made to the source directory will be used.

```
conda create -n aimm python=3.9
conda activate aimm
git clone git@github.com:AI-multimodal/aimmdb.git
pip install -e ./aimmdb
```

## Mongo

aimmdb relies on [mongodb](https://www.mongodb.com/) for storing and searching metadata.
Therefore in order to run aimmdb we must have an accessible mongo instance also running.
For convenience we will run mongo in a docker container.
We secure our mongo instance with a password stored in the `MONGO_PASSWORD` environment variable.
The local [configuration](/deploy/local/config.yml) references this environment variable in order to provide aimmdb authenticated access to the mongo instance.

```
export MONGO_PASSWORD=SECRET
docker run --rm --name mongodb -p 27018:27017 -e MONGO_INITDB_ROOT_USERNAME=root -e MONGO_INITDB_ROOT_PASSWORD=$MONGO_PASSWORD -d mongo
```

## Starting the Server

Now we are almost ready to start the tiled server.
But first we must create a directory for aimmdb to store its data in.

```
# navigate to the directory with the local configuration
cd deploy/local

# create the data directory
mkdir data
```

Now we can start the server

```
tiled serve config config.yml
```

## Testing

The server should now be accessible on localhost.
We can test it by connecting a python client to the server

```python
from tiled.client import from_uri
c = from_uri("http://localhost:8000/api")
```

Note this will prompt for a username and password which are defined directly in the [configuration](/deploy/local/config.yml) in this toy example.

Now that the server is running we can ingest and search data as shown [here](/ingest/ingest_newville_example.ipynb).
