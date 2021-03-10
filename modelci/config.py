import os

# MongoDB configuration
MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
MONGO_PORT = int(os.getenv('MONGO_PORT', 27017))
MONGO_USERNAME = os.getenv('MONGO_USERNAME', 'modelci')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD', 'modelci@2020')
MONGO_DB = os.getenv('MONGO_DB', 'modelci')
MONGO_AUTH_SOURCE = os.getenv('MONGO_AUTH_SOURCE', 'modelci')

# cAdvisor configuration
CADVISOR_PORT = int(os.getenv('CADVISOR_PORT', 8080))

# Node exporter configuration
NODE_EXPORTER_PORT = int(os.getenv('NODE_EXPORTER_PORT', 9400))
