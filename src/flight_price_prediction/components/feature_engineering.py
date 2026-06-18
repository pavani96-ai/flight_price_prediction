import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.flight_price_prediction.entity.config_entity import FeatureEngineeringConfig
from src.flight_price_prediction.entity.artifact_entity import DataValidationArtifact, FeatureEngineeringArtifact
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.utils.common import save_data,save_bin, create_directory
from src.flight_price_prediction.utils.feature_engineering_utils import *





class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.transformer = ColumnTransformer(
            transformers=[
                ('airline', 'passthrough', ['Airline']),
                ('source', 'passthrough', ['Source']),
                ('destination', DestinationNormalizer(self.config.replace_destinations), ['Destination']),
                ('total_stops', TotalStopsTransformer(self.config.stop_map), ['Total_Stops']),
                ('route_interaction', RouteInteractionTransformer(self.config.replace_destinations), ['Source', 'Destination']),
                ('date', DateFeatureExtractor(), ['Date_of_Journey']),
                ('duration', DurationMinutesTransformer(), ['Duration']),
                ('dep_time', TimeCyclicTransformer(), ['Dep_Time']),
                ('arrival_time', TimeCyclicTransformer(), ['Arrival_Time']),
                ('duration_times_stops', DurationStopsInteractionTransformer(), ['Duration', 'Total_Stops']),
            ],
            sparse_threshold=0,
        
            remainder='drop',)

    def fit(self, X, y=None):
        self.transformer.fit(X)
        return self

    def transform(self, X):
        transformed = self.transformer.transform(X)
        columns = [
            'airline',
            'source',
            'destination',
            'total_stops',
            'route_interaction',
            'month',
            'day',
            'duration_to_minutes',
            'dep_sin',
            'dep_cos',
            'arrival_sin',
            'arrival_cos',
            'duration_times_stops',
        ]
        df = pd.DataFrame(transformed, columns=columns)
        return df


class FeatureEngineering:
    def __init__(self,
                 feature_engineering_config: FeatureEngineeringConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.config = feature_engineering_config
            self.data_validation_artifact = data_validation_artifact
            self.feature_transformer = FeatureEngineeringTransformer(self.config)
        except Exception as e:
            raise CustomException(e, sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    def _clean_raw_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.rename(columns={
            'Airline': 'Airline',
            'Date_of_Journey': 'Date_of_Journey',
            'Source': 'Source',
            'Destination': 'Destination',
            'Route': 'Route',
            'Dep_Time': 'Dep_Time',
            'Arrival_Time': 'Arrival_Time',
            'Duration': 'Duration',
            'Total_Stops': 'Total_Stops',
            'Additional_Info': 'Additional_Info',
            'Price': 'Price'
        })
        dataframe = dataframe[dataframe['Duration'] != '5m']
        dataframe = dataframe.drop_duplicates()
        return dataframe

    def initiate_feature_engineering(self) -> FeatureEngineeringArtifact:
        try:
            logging.info('Starting feature engineering on validated datasets')
            train_df = FeatureEngineering.read_data(self.data_validation_artifact.valid_train_file_name)
            test_df = FeatureEngineering.read_data(self.data_validation_artifact.valid_test_file_name)
            
            logging.info('cleaning data and renaming columns started')
            train_df = self._clean_raw_dataframe(train_df)
            test_df = self._clean_raw_dataframe(test_df)
            
            logging.info('dividing target column as train_target ans test_target before performing featuire engineering')
            train_target = pd.to_numeric(train_df['Price'], errors='coerce')
            test_target = pd.to_numeric(test_df['Price'], errors='coerce')

            logging.info("apllying feature_transformer for train and test sets")
            train_engineered = self.feature_transformer.fit_transform(train_df)
            test_engineered = self.feature_transformer.transform(test_df)
            
            logging.info("Saving Feature Engineering Transformer as .pkl")
            feature_engineered_pkl_file = Path(self.config.feature_engineered_pkl_file) / "feature_engineering.pkl"
            save_bin(self.feature_transformer, feature_engineered_pkl_file )
            create_directory("final_model/feature_engineering.pkl")
            save_bin(self.feature_transformer, "final_model/feature_engineering.pkl")
            logging.info("Preprocessor saved to final_model/feature_engineering.pkl")

            logging.info('adding target column again to train and test engineered sets')
            train_engineered['price'] = train_target.values
            test_engineered['price'] = test_target.values
            
            logging.info('Dropping null values')
            train_engineered = train_engineered.dropna(subset=['price'])
            test_engineered = test_engineered.dropna(subset=['price'])

            logging.info("saving engineered train and test data into artifacts")
            save_data(train_engineered, Path(self.config.engineered_train_file_name))
            save_data(test_engineered, Path(self.config.engineered_test_file_name))
            
            logging.info(f'Feature engineered train file saved to {self.config.engineered_train_file_name}')
            logging.info(f'Feature engineered test file saved to {self.config.engineered_test_file_name}')

            return FeatureEngineeringArtifact(
                engineered_train_file_name=Path(self.config.engineered_train_file_name),
                engineered_test_file_name=Path(self.config.engineered_test_file_name),
            )
        except Exception as e:
            raise CustomException(e, sys)


             




