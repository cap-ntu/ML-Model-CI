import os

# MongoDB configuration
MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
MONGO_PORT = int(os.getenv('MONGO_PORT', 27017))
MONGO_USERNAME = os.getenv('MONGO_USERNAME', 'modelci')
MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
MONGO_DB = os.getenv('MONGO_DB', 'modelci')
MONGO_AUTH_SOURCE = os.getenv('MONGO_AUTH_SOURCE', 'modelci')
MONGO_CONTAINER_LABEL = os.getenv('MONGO_CONTAINER_LABEL', 'modelci.mongo')

# cAdvisor configuration
CADVISOR_PORT = int(os.getenv('CADVISOR_PORT', 8080))
CADVISOR_CONTAINER_LABEL = os.getenv('CADVISOR_CONTAINER_LABEL', 'modelci.cadvisor')

# Node exporter configuration
NODE_EXPORTER_PORT = int(os.getenv('NODE_EXPORTER_PORT', 9400))
DCGM_EXPORTER_CONTAINER_LABEL = os.getenv('DCGM_EXPORTER_CONTAINER_LABEL', 'modelci.dcgm-exporter')
GPU_METRICS_EXPORTER_CONTAINER_LABEL = os.getenv('GPU_METRICS_EXPORTER_CONTAINER_LABEL', 'modelci.gpu-metrics-exporter')
