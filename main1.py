from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.pipeline.train_pipeline import TrainPipeline
import sys
import os, dagshub

user = os.getenv("DAGSHUB_USER")
token = os.getenv("DAGSHUB_TOKEN")

if not user or not token:
    raise RuntimeError("DAGSHUB_USER or DAGSHUB_TOKEN not found in environment")

dagshub.auth.add_app_token(token)
dagshub.init(repo_owner=user, repo_name="flight_price_prediction", mlflow=True)

STAGE_NAME = " TRAINING PIPELINE"

if __name__ == "__main__":
    try:
        logging.info(f">>>> stage {STAGE_NAME} started <<<<")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        logging.info(f">>>> stage {STAGE_NAME} completed <<<<")
    except Exception as e:
        raise CustomException(e, sys)