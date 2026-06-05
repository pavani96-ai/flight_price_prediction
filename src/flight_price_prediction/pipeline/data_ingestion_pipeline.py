from src.flight_price_prediction.config.configuration import ConfigurationManager
from src.flight_price_prediction.components.data_ingestion import DataIngestion
from src.flight_price_prediction.logging.logger import logger
from src.flight_price_prediction.exception.exception import CustomException
import sys
STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self, config: ConfigurationManager):
        try:
            self.config = config
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        config = ConfigurationManager()
        obj = DataIngestionTrainingPipeline(config =config)
        obj.initiate_data_ingestion()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx====x")
    except Exception as e:
        logger.exception(e)
        raise e
#finidhed update