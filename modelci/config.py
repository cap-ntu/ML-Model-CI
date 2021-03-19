import os
from pathlib import Path

from pydantic import BaseSettings, SecretStr, SecretBytes

# API
API_V1_STR = '/api/v1'
API_EXP_STR = '/api/exp'


class DBSettings(BaseSettings):
    mongo_host: str = 'localhost'
    mongo_port: int = 27017
    mongo_username: str = 'modelci'
    mongo_password: SecretStr = SecretStr('modelci@2020')
    mongo_db: str = 'modelci'
    mongo_auth_source: str = 'modelci'
    auth_mechanism: str = 'SCRAM-SHA-256'

    class Config:
        env_file = Path(__file__).absolute().parent / '.env'


class ServiceSettings(BaseSettings):
    mongo_host: str = 'localhost'
    mongo_port: int = 27017

    # cAdvisor configuration
    cadvisor_port: int = 8080

    # Node exporter configuration
    node_exporter_port: int = 9400

    class Config:
        env_file = Path(__file__).absolute().parent / '.env'


class AppSettings(BaseSettings):
    project_name: str = 'ModelCI'
    backend_cors_origins: str = '*'
    server_host: str = 'localhost'
    server_port: int = 8000
    secret_key: SecretBytes = SecretBytes(os.urandom(32))
    access_token_expire_minutes: int = 60 * 24 * 8  # 60 minutes * 24 hours * 8 days = 8 days

    class Config:
        env_file = Path(__file__).absolute().parent / '.env'

    @property
    def server_url(self):
        return f'http://{self.server_host}:{self.server_port}'

    @property
    def api_v1_prefix(self):
        return f'http://{self.server_host}:{self.server_port}{API_V1_STR}'


service_settings = ServiceSettings()
db_settings = DBSettings()
app_settings = AppSettings()
