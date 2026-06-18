from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.pipeline.train_pipeline import TrainPipeline
import sys

STAGE_NAME = " TRAINING PIPELINE"

if __name__ == "__main__":
    try:
        logging.info(f">>>> stage {STAGE_NAME} started <<<<")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logging.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        raise CustomException(e, sys)