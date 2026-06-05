import sys
from src.flight_price_prediction.config.configuration import ConfigurationManager
from src.flight_price_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.flight_price_prediction.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging

def main():
    try:
        config = ConfigurationManager()
        logging.info(">>>> Data Ingestion Stage Started <<<<")
        data_ingestion_pipeline = DataIngestionTrainingPipeline(config =config)
        ingestion_artifact = data_ingestion_pipeline.initiate_data_ingestion()
        logging.info(">>>> Data Ingestion Stage Completed <<<<\n\nx====x")
    
        logging.info(">>>> Data Validation Stage Started <<<<")
        data_validation_pipeline = DataValidationTrainingPipeline(config =config,data_ingestion_artifact=ingestion_artifact)
        validation_artifact = data_validation_pipeline.initiate_data_validation()
        logging.info(">>>> Data Validation Stage Completed <<<<\n\nx====x")


    except Exception as e:
        raise CustomException(e, sys)
    

if __name__ == "__main__":
    main()
