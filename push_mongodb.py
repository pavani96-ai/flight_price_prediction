import os
import sys
import json

from requests import head
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.server_api import ServerApi

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
load_dotenv()


MONGO_DB_URL = os.getenv("MONGO_DB_URL")

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging

class MongoDBClient:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)
        
    def xlsx_to_json(self,filepath):
        try:
            data = pd.read_excel(filepath)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e, sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(records)
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    FILE_PATH = "data/Flight_Fare.xlsx"
    DATABASE = "Flight_Price_Prediction"
    COLLECTION = "Flight_price_prediction"

    try:
        logging.info("Extracting data from xlsx file")
        data_extractor = MongoDBClient()
        records = data_extractor.xlsx_to_json(FILE_PATH)

        logging.info(f"Converted {len(records)} records. Pushing to MongoDB...")
        no_of_records = data_extractor.insert_data_mongodb(
            records =records,
            database = DATABASE,
            collection=COLLECTION
        )
        print(f"Success! Inserted {no_of_records} records into MongoDB")

    except Exception as e:
        print(f"an error occurred during execution: {e}")
