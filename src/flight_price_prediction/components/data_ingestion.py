import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pymongo
from typing import List
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.entity.config_entity import DataIngestionConfig
from src.flight_price_prediction.entity.artifact_entity import DataIngestionArtifact
from src.flight_price_prediction.utils.common import *
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

MONGO_DB_URL = os.getenv("MONGO_DB_URL", "mongodb://localhost:27017")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """ 
         Initiates the DataIngestion component.
         Ensure your MONGO_DB_URL is set in your environment variables
         
         """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.mongo_db_url = os.getenv("MONGO_DB_URL")
            self.mongo_client = pymongo.MongoClient(self.mongo_db_url)
            # CRITICAL: Validate that it exists!
            if not self.mongo_db_url:
                raise ValueError("Environment variable MONGO_DB_URL is not set or empty.")
            
            logging.info(f"Connecting to MongoDB at: {self.mongo_db_url[:15]}...") # Log first 15 chars
            self.mongo_client = pymongo.MongoClient(self.mongo_db_url)
            
            # Force a ping to verify connectivity immediately
            self.mongo_client.admin.command('ping')
            logging.info("Successfully connected to MongoDB.")
        except Exception as e:
            raise CustomException(e, sys)

    def export_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_store_file_name = Path(self.data_ingestion_config.feature_store_file_name)
            create_directory(feature_store_file_name)
            save_data(dataframe, feature_store_file_name)
            logging.info(f"saved raw data into feature store file path: {feature_store_file_name}")
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logging.info("splitting data into train_test_split_ratio")
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
            )

            training_file_name= Path(self.data_ingestion_config.training_file_name)
            create_directory(training_file_name)
            save_data(train_set, training_file_name)

            testing_file_name = Path(self.data_ingestion_config.testing_file_name)
            create_directory(testing_file_name)
            save_data(test_set, testing_file_name)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                training_file_name=Path(self.data_ingestion_config.training_file_name),
                testing_file_name=Path(self.data_ingestion_config.testing_file_name),
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)



        