import mongoengine as mongo


class MongoDB(object):
    """MongoDB connection manager.

    This class manages the connections made to MongoDB once the connection settings are given. Connections are held
    and could be closed given connection alias.
    """
    def __init__(self, db: str = None, *, host: str = None, port: int = None, username: str = None,
                 password: str = None,
                 auth_source: str = 'admin', **kwargs):
        """Create a MongoDB connection manager.

        Args:
            db (:obj:`str`, optional): Database name. Default to None.
            host (:obj:`str`, optional): MongoDB host address. Default to None.
            port (:obj:`str`, optional): MongoDB port address. Default to None.
            username (:obj:`str`, optional): Username. Default to None.
            password (:obj:`str`, optional): Password. Default to None.
            auth_source (:obj:`str`, optional): Authentication source database. Default to 'admin'.
        """
        self._conn_settings = {
            'name': db,
            'host': host,
            'port': port,
            'username': username,
            'password': password,
            'authentication_source': auth_source,
            **kwargs
        }
        self._sessions = []

        self.db = None

    def connect(self, alias='default'):
        """Connect to a MongoDB session.

        Args:
             alias (:obj:`str`, optional): The alias name. Default to 'default'.
        """
        self.db = mongo.connect(alias=alias, **self._conn_settings)
        self._sessions.append(alias)

    def close(self, alias: str = None):
        """Close a connection, given alias name.

        Args:
            alias (:obj:`str`, optional): The alias name. Default to None.
                When alias name is provided, connection with the alias name is closed. Otherwise, it closes all the
                tracked connections.
        """
        if alias in self._sessions:
            mongo.disconnect(alias)
        if alias is None:
            for alias in self._sessions:
                mongo.disconnect(alias)
