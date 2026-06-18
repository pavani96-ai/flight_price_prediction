from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.entity.config_entity import ModelTrainerConfig
from src.flight_price_prediction.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from src.flight_price_prediction.config.configuration import ConfigurationManager
from src.flight_price_prediction.utils.metrics.regression_metrics import get_regression_score
from src.flight_price_prediction.utils.common import load_bin, save_bin, evaluate_models, create_directory
import sys
import os
import joblib
from src.flight_price_prediction.utils.ml_utils.model import Flight_Price_Prediction_Model
#model trainer component
from dotenv import load_dotenv

load_dotenv()
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

import mlflow
from urllib.parse import urlparse

import dagshub
token = os.getenv("DAGSHUB_TOKEN")

# 2. Login explicitly if the token exists
if token:
    dagshub.login(token=token)
else:
    # This acts as a fallback or triggers the auth requirement error
    print("WARNING: DAGSHUB_TOKEN not found in environment.")

dagshub.init(repo_owner = 'pavani96-ai', repo_name = 'flight_price_prediction', mlflow=True)

os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def track_mlflow(self, best_model, regressionmetric,best_model_name):
        mlflow.set_registry_uri(os.getenv('MLFLOW_TRACKING_URI'))
        tracking_url_type_store = urlparse(os.getenv('MLFLOW_TRACKING_URI')).scheme
        with mlflow.start_run():
            r2_score = regressionmetric.r2_score
            mean_absolute_error = regressionmetric.mean_absolute_error
            mean_squared_error = regressionmetric.mean_squared_error
            adj_r2_score = regressionmetric.adj_r2_score

            mlflow.log_metric('r2_score', r2_score)
            mlflow.log_metric('mean_absolute_error', mean_absolute_error)
            mlflow.log_metric('mean_squared_error', mean_squared_error)
            mlflow.log_metric('adj_r2_score', adj_r2_score)

            if tracking_url_type_store != 'file':
                mlflow.sklearn.log_model(best_model, 'model', registered_model_name=best_model_name)
            else:
                mlflow.sklearn.log_model(best_model, 'model')
    
    def train_model(self,x_train,y_train,x_test,y_test):
        try:
            models = {
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'XGBoost Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=0),
                'LightGBM Regressor': LGBMRegressor(verbose=-1),
             }
            params = {
                'Decision Tree Regressor': {
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'criterion': ['squared_error', 'friedman_mse'],
                    },
                'Random Forest Regressor': {
                    'n_estimators': [32, 64, 128],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'max_features': ['sqrt', 'log2'],
                    },
                'Gradient Boosting Regressor': {
                    'n_estimators': [32, 64, 128],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0],
                    },
                'XGBoost Regressor': {
                    'n_estimators': [32, 64, 128],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    },
                'CatBoost Regressor': {
                    'iterations': [50, 100],
                    'learning_rate': [0.05, 0.1],
                    'depth': [6, 8],
                    },
                'LightGBM Regressor': {
                    'n_estimators': [32, 64, 128],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 50],
                    'max_depth': [5, 10],
                    },
            }
            model_report:dict=evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param =params)

            best_model_name = max(
                model_report,
                key=lambda name: model_report[name]["test"]["adj_r2"],
            )
            best_model_score = model_report[best_model_name]["test"]["adj_r2"]
            best_model = models[best_model_name]
            logging.info(f"Best model selected: {best_model_name} (test adj_r2={best_model_score:.4f})")
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)
            
            #calculationg regression metrics for train set and tracking with mlflow
            regression_train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred, n_features = x_train.shape[1])
            self.track_mlflow(best_model, regression_train_metric, best_model_name)
            logging.info(f"Best model found on training set: {best_model_name} )")

            #claculating regression metrics for test set and tracking with mlflow
            regression_test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred, n_features = x_test.shape[1])
            self.track_mlflow(best_model, regression_test_metric, best_model_name)
            logging.info(f"Best Model found on test set: {best_model_name})") 

            preprocessor = joblib.load(self.data_transformation_artifact.preprocessor_object_file_name)
            Flight_Fare_Prediction_Model = Flight_Price_Prediction_Model(preprocessor=preprocessor, model=best_model)
            save_bin(Flight_Fare_Prediction_Model, self.model_trainer_config.model_file_name)

            save_bin(best_model, "final_model/model.pkl")
            create_directory("final_model/preprocessor.pkl")
            save_bin(preprocessor, "final_model/preprocessor.pkl")
            logging.info("Preprocessor saved to final_model/preprocessor.pkl")

            #3 model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(model_file_name = self.model_trainer_config.model_file_name,
                                                          train_metric_artifact = regression_train_metric,
                                                          test_metric_artifact = regression_test_metric)
        
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info('loading transformed training and test data')
            train_file_name = self.data_transformation_artifact.transformed_train_file_name
            test_file_name = self.data_transformation_artifact.transformed_test_file_name

            #loading training array annd testing array
            train_arr = load_bin(path = train_file_name)
            test_arr = load_bin(path = test_file_name)

            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1])
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        