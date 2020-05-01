from modelci import config
from .mongo_db import MongoDB

conn_settings = {
    'db': config.MONGO_DB,
    'host': config.MONGO_HOST,
    'port': int(config.MONGO_PORT),
    'username': config.MONGO_USERNAME,
    'password': config.MONGO_PASSWORD,
    'authentication_source': config.MONGO_AUTH_SOURCE
}

mongo = MongoDB(**conn_settings)

mongo.connect()
