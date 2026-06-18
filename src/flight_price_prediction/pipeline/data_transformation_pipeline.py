import sys
from src.flight_price_prediction.config.configuration import ConfigurationManager
from src.flight_price_prediction.components.data_transformation import DataTransformation
from src.flight_price_prediction.entity.artifact_entity import FeatureEngineeringArtifact, DataTransformationArtifact
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self, config: ConfigurationManager, feature_engineering_artifact: FeatureEngineeringArtifact):
        try:
            self.config = config
            self.feature_engineering_artifact = feature_engineering_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f'>>>> stage {STAGE_NAME} started <<<<')
            data_transformation_config = self.config.get_data_transformation_config()
            data_transformation = DataTransformation(
                data_transformation_config=data_transformation_config,
                feature_engineering_artifact=self.feature_engineering_artifact,
            )
            artifact = data_transformation.initiate_data_transformation()
            logging.info(f'>>>> stage {STAGE_NAME} completed <<<<')
            return artifact
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    try:
        logging.info(f'>>>> stage {STAGE_NAME} started <<<<')
        config = ConfigurationManager()
        from src.flight_price_prediction.pipeline.feature_engineering_pipeline import FeatureEngineeringTrainingPipeline
        from src.flight_price_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
        from src.flight_price_prediction.pipeline.data_validation_pipeline import DataValidationTrainingPipeline

        ingestion_pipeline = DataIngestionTrainingPipeline(config=config)
        ingestion_artifact = ingestion_pipeline.initiate_data_ingestion()
        validation_pipeline = DataValidationTrainingPipeline(config=config, data_ingestion_artifact=ingestion_artifact)
        validation_artifact = validation_pipeline.initiate_data_validation()
        feature_engineering_pipeline = FeatureEngineeringTrainingPipeline(config=config, data_validation_artifact=validation_artifact)
        fe_artifact = feature_engineering_pipeline.initiate_feature_engineering()
        pipeline = DataTransformationTrainingPipeline(config=config, feature_engineering_artifact=fe_artifact)
        pipeline.initiate_data_transformation()
        logging.info(f'>>>> stage {STAGE_NAME} completed <<<<')
    except Exception as e:
        raise CustomException(e, sys)
