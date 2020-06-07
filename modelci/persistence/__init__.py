from modelci import config
from .mongo_db import MongoDB

conn_settings = {
    'db': str(config.MONGO_DB),
    'host': str(config.MONGO_HOST),
    'port': int(config.MONGO_PORT),
    'username': str(config.MONGO_USERNAME),
    'password': str(config.MONGO_PASSWORD),
    'authentication_source': str(config.MONGO_AUTH_SOURCE)
}

mongo = MongoDB(**conn_settings)

mongo.connect()
