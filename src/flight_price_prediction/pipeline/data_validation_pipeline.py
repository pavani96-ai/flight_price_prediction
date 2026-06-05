#data validation pipeline
import sys
from src.flight_price_prediction.config.configuration import ConfigurationManager
from src.flight_price_prediction.entity.artifact_entity import DataIngestionArtifact
from src.flight_price_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.flight_price_prediction.components.data_validation import DataValidation
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.exception.exception import CustomException



STAGE = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self, config: ConfigurationManager,data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.config = config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_validation(self):
        try:
            data_validation_config = self.config.get_data_validation_config()
            data_validation = DataValidation(
                data_ingestion_artifact = self.data_ingestion_artifact,
                data_validation_config = data_validation_config
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    try:
        logging.info(f">>>> stage {STAGE} started <<<<")
        config = ConfigurationManager()
        ingestion_pipeline = DataIngestionTrainingPipeline(config=config)
        data_ingestion_artifact = ingestion_pipeline.initiate_data_ingestion()
        obj = DataValidationTrainingPipeline(config =config, data_ingestion_artifact= data_ingestion_artifact)
        obj.initiate_data_validation()
        logging.info(f">>>> stage {STAGE} completed <<<<")
    except Exception as e:
        raise CustomException(e, sys)