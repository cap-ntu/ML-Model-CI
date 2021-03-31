from modelci.config import db_settings
from .mongo_db import MongoDB

conn_settings = {
    'db': str(db_settings.mongo_db),
    'host': str(db_settings.mongo_host),
    'port': int(db_settings.mongo_port),
    'username': str(db_settings.mongo_username),
    'password': db_settings.mongo_password.get_secret_value(),
    'authentication_source': str(db_settings.mongo_auth_source)
}

mongo = MongoDB(**conn_settings)

mongo.connect()
