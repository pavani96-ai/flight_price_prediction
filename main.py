import sys
from src.flight_price_prediction.config.configuration import ConfigurationManager
from src.flight_price_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.flight_price_prediction.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.flight_price_prediction.pipeline.feature_engineering_pipeline import FeatureEngineeringTrainingPipeline
from src.flight_price_prediction.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.flight_price_prediction.pipeline.model_trainer_pipeline import ModelTrainerPipeline

from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging

def main():
    try:
        config = ConfigurationManager()
        logging.info(">>>> Data Ingestion Stage Started <<<<")
        data_ingestion_pipeline = DataIngestionTrainingPipeline(config=config)
        ingestion_artifact = data_ingestion_pipeline.initiate_data_ingestion()
        logging.info(">>>> Data Ingestion Stage Completed <<<<")
        logging.info("x====x")
    
        logging.info(">>>> Data Validation Stage Started <<<<")
        data_validation_pipeline = DataValidationTrainingPipeline(config=config, data_ingestion_artifact=ingestion_artifact)
        validation_artifact = data_validation_pipeline.initiate_data_validation()
        logging.info(">>>> Data Validation Stage Completed <<<<")
        logging.info("x====x")

        logging.info(">>>> Feature Engineering Stage Started <<<<")
        feature_engineering_pipeline = FeatureEngineeringTrainingPipeline(config=config, data_validation_artifact=validation_artifact)
        feature_engineering_artifact = feature_engineering_pipeline.initiate_feature_engineering()
        logging.info(">>>> Feature Engineering Stage Completed <<<<")
        logging.info("x====x")

        logging.info(">>>> Data Transformation Stage Started <<<<")
        data_transformation_pipeline = DataTransformationTrainingPipeline(config=config, feature_engineering_artifact=feature_engineering_artifact)
        data_transformation_artifact = data_transformation_pipeline.initiate_data_transformation()
        logging.info(">>>> Data Transformation Stage Completed <<<<")
        logging.info("x====x")

        logging.info(">>>> Model Trainer Stage Started <<<<")
        model_trainer_pipeline = ModelTrainerPipeline(
            config=config,
            data_transformation_artifact=data_transformation_artifact,
        )
        model_trainer_artifact = model_trainer_pipeline.initiate_model_training()
        logging.info(">>>> Model Trainer Stage Completed <<<<")
        logging.info("x====x")


    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
