from src.flight_price_prediction.constants import *
from src.flight_price_prediction.utils.common import read_yaml, create_directories
from src.flight_price_prediction.entity.config_entity import DataIngestionConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            feature_store_file_name=Path(config.feature_store_file_name),
            training_file_name=Path(config.training_file_name),
            testing_file_name=Path(config.testing_file_name),
            train_test_split_ratio = params.train_test_split_ratio,
            collection_name=config.collection_name,
            database_name=config.database_name
        )

        return data_ingestion_config

