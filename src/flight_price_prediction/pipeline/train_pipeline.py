import os
import sys
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.components.data_ingestion import DataIngestion
from src.flight_price_prediction.components.data_validation import DataValidation
from src.flight_price_prediction.components.feature_engineering import FeatureEngineering
from src.flight_price_prediction.components.data_transformation import DataTransformation
from src.flight_price_prediction.components.model_training import ModelTrainer

from src.flight_price_prediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    FeatureEngineeringArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from src.flight_price_prediction.cloud.s3_syncer import S3Sync

from src.flight_price_prediction.config.configuration import ConfigurationManager

STAGE_NAME = "Train Pipeline"

class TrainPipeline:
    def __init__(self):
        self.s3_sync = S3Sync()
        self.config_manager = ConfigurationManager()
        self.s3_config = self.config_manager.get_s3_sync_config()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            logging.info('start data ingestion')
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info('data_ingestion_completed')
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info('data_validation_completed')
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_feature_engineering(self, data_validation_artifact: DataValidationArtifact) -> FeatureEngineeringArtifact:
        try:
            feature_engineering_config = self.config_manager.get_feature_engineering_config()
            feature_engineering = FeatureEngineering(data_validation_artifact = data_validation_artifact,feature_engineering_config=feature_engineering_config)
            feature_engineering_artifact = feature_engineering.initiate_feature_engineering()
            logging.info('feature engineering completed')
            return feature_engineering_artifact

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, feature_engineering_artifact: FeatureEngineeringArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(feature_engineering_artifact=feature_engineering_artifact,data_transformation_config = data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info('data_transformation_completed')
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = self.config_manager.get_model_training_config()
            model_trainer =ModelTrainer(data_transformation_artifact =data_transformation_artifact,model_trainer_config =model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info('model_training_completed')
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{self.s3_config.training_bucket_name}"
            self.s3_sync.sync_folder_to_s3(
                folder=str(self.s3_config.artifact_dir),
                aws_bucket_url=aws_bucket_url,
            )
            logging.info(f"artifacts has been synced to {aws_bucket_url}")
        except Exception as e:
            raise CustomException(e, sys)
    def sync_final_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{self.s3_config.training_bucket_name}"
            self.s3_sync.sync_folder_to_s3(
                folder=str(self.s3_config.final_model_dir),
                aws_bucket_url=aws_bucket_url,
            )
            logging.info(f"final model has been synced to {aws_bucket_url}")
        except Exception as e:
            raise CustomException(e, sys)
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            feature_engineering_artifact = self.start_feature_engineering(data_validation_artifact=data_validation_artifact)
            data_transformation_artifact=self.start_data_transformation(feature_engineering_artifact=feature_engineering_artifact)
            model_trainer_artifact=self.start_model_training(data_transformation_artifact=data_transformation_artifact)
            
            self.sync_artifact_dir_to_s3()
            self.sync_final_model_dir_to_s3()
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    try:
        logging.info(f">>>> stage {STAGE_NAME} started <<<<")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logging.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        raise CustomException(e, sys)