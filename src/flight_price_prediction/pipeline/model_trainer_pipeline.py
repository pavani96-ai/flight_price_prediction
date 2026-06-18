from src.flight_price_prediction.config.configuration import ConfigurationManager
from src.flight_price_prediction.entity.config_entity import ModelTrainerConfig
from src.flight_price_prediction.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
import sys
from src.flight_price_prediction.components.model_training import ModelTrainer


from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging

STAGE_NAME = "Model Trainer Stage"

class ModelTrainerPipeline:
    def __init__(self, config: ConfigurationManager, data_transformation_artifact: DataTransformationArtifact):
       try:
           self.config = config
           self.data_transformation_artifact = data_transformation_artifact
       except Exception as e:
           raise CustomException(e, sys)
       
    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info(f">>>> stage {STAGE_NAME} started <<<<")
            model_trainer_config = self.config.get_model_training_config()
            model_trainer = ModelTrainer(model_trainer_config = model_trainer_config, data_transformation_artifact = self.data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f">>>> stage {STAGE_NAME} completed <<<<")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    try:
        logging.info(f'>>>> stage {STAGE_NAME} started <<<<')
        config = ConfigurationManager()
        from src.flight_price_prediction.pipeline.feature_engineering_pipeline import FeatureEngineeringTrainingPipeline
        from src.flight_price_prediction.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
        from src.flight_price_prediction.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
        from src.flight_price_prediction.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline

        ingestion_pipeline = DataIngestionTrainingPipeline(config=config)
        ingestion_artifact = ingestion_pipeline.initiate_data_ingestion()
        validation_pipeline = DataValidationTrainingPipeline(config=config, data_ingestion_artifact=ingestion_artifact)
        validation_artifact = validation_pipeline.initiate_data_validation()
        feature_engineering_pipeline = FeatureEngineeringTrainingPipeline(config=config, data_validation_artifact=validation_artifact)
        fe_artifact = feature_engineering_pipeline.initiate_feature_engineering()
        transformation_pipeline = DataTransformationTrainingPipeline(config=config, feature_engineering_artifact=fe_artifact)
        transformation_pipeline.initiate_data_transformation()
        model_trainer_pipeline = ModelTrainerPipeline(config=config, data_transformation_artifact=transformation_pipeline.initiate_data_transformation())
        model_trainer_artifact = model_trainer_pipeline.initiate_model_training()
        logging.info(f'>>>> stage {STAGE_NAME} completed <<<<')
    except Exception as e:
        raise CustomException(e, sys)