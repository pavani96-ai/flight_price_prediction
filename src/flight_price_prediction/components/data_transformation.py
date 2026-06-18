import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.flight_price_prediction.entity.config_entity import DataTransformationConfig
from src.flight_price_prediction.entity.artifact_entity import FeatureEngineeringArtifact, DataTransformationArtifact
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.utils.common import save_bin


class DataTransformation:
    def __init__(self,
                 data_transformation_config: DataTransformationConfig,
                 feature_engineering_artifact: FeatureEngineeringArtifact):
        try:
            self.config = data_transformation_config
            self.feature_engineering_artifact = feature_engineering_artifact
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def _build_transformer(self) -> Pipeline:
        numeric_transformer = self.config.scaler
        categorical_transformer = self.config.encoder
        try:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numeric' , numeric_transformer, self.config.numerical_columns),
                    ('categorical', categorical_transformer, self.config.categorical_columns),
                ],
                remainder ='drop',
            )

            pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info('Starting data transformation on engineered datasets')

            train_df = pd.read_csv(self.feature_engineering_artifact.engineered_train_file_name)
            test_df = pd.read_csv(self.feature_engineering_artifact.engineered_test_file_name)

            target_column = self.config.target_column
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column].values
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column].values

            transformer = self._build_transformer()
            X_train_transformed = transformer.fit_transform(X_train)
            X_test_transformed = transformer.transform(X_test)

            train_array = np.c_[X_train_transformed, y_train]
            test_array = np.c_[X_test_transformed, y_test]

            save_bin(train_array, self.config.transformed_train_file_name)
            save_bin(test_array, self.config.transformed_test_file_name)
            joblib.dump(transformer, self.config.preprocessor_object_file_name)

            logging.info(f'Transformed train data saved to {self.config.transformed_train_file_name}')
            logging.info(f'Transformed test data saved to {self.config.transformed_test_file_name}')
            logging.info(f'Preprocessor object saved to {self.config.preprocessor_object_file_name}')

            return DataTransformationArtifact(
                transformed_train_file_name=self.config.transformed_train_file_name,
                transformed_test_file_name=self.config.transformed_test_file_name,
                preprocessor_object_file_name=self.config.preprocessor_object_file_name,
            )
        except Exception as e:
            raise CustomException(e, sys)
