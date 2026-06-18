import sys
from src.flight_price_prediction.config.configuration import ConfigurationManager
from src.flight_price_prediction.components.feature_engineering import FeatureEngineering
from src.flight_price_prediction.entity.artifact_entity import DataValidationArtifact, FeatureEngineeringArtifact
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging

STAGE_NAME = "Feature Engineering Stage"

class FeatureEngineeringTrainingPipeline:
    def __init__(self, config: ConfigurationManager, data_validation_artifact: DataValidationArtifact):
        try:
            self.config = config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        try:
            logging.info(f'>>>> stage {STAGE_NAME} started <<<<')
            feature_engineering_config = self.config.get_feature_engineering_config()
            feature_engineering = FeatureEngineering(
                feature_engineering_config=feature_engineering_config,
                data_validation_artifact=self.data_validation_artifact,
            )
            artifact = feature_engineering.initiate_feature_engineering()
            logging.info(f'>>>> stage {STAGE_NAME} completed <<<<')
            return artifact
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    try:
        logging.info(f'>>>> stage {STAGE_NAME} started <<<<')
        config = ConfigurationManager()
        from src.flight_price_prediction.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
        from src.flight_price_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
        ingestion_pipeline = DataIngestionTrainingPipeline(config=config)
        ingestion_artifact = ingestion_pipeline.initiate_data_ingestion()
        validation_pipeline = DataValidationTrainingPipeline(config=config, data_ingestion_artifact=ingestion_artifact)
        validation_artifact = validation_pipeline.initiate_data_validation()
        pipeline = FeatureEngineeringTrainingPipeline(config=config, data_validation_artifact=validation_artifact)
        pipeline.initiate_feature_engineering()
        logging.info(f'>>>> stage {STAGE_NAME} completed <<<<')
    except Exception as e:
        raise CustomException(e, sys)
