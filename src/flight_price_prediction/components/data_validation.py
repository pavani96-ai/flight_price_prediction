#data validation component
import sys
from pathlib import Path
from src.flight_price_prediction.entity.config_entity import DataValidationConfig
from src.flight_price_prediction.entity.artifact_entity import DataIngestionArtifact,DatavalidationArtifact
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.utils.common import *
from scipy.stats import chi2_contingency
import pandas as pd

class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = self.data_validation_config.all_schema
        except Exception as e:
            raise CustomException(e, sys)
    
    @staticmethod  
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
    
    def impute_null_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Impute null values using mode imputation"""
        try:
            if dataframe.isnull().sum().sum() == 0:
                logging.info("No null values found in dataframe")
                return dataframe
            
            logging.info("Null values detected. Imputing using mode")
            
            # Create a copy to avoid modifying original
            df_imputed = dataframe.copy()
            
            # Fill all columns with mode
            for col in df_imputed.columns:
                if df_imputed[col].isnull().sum() > 0:
                    mode_value = df_imputed[col].mode()[0] if len(df_imputed[col].mode()) > 0 else 'Unknown'
                    df_imputed[col] = df_imputed[col].fillna(mode_value)
                    logging.info(f"Filled null values in {col} with mode: {mode_value}")
            
            logging.info("All null values have been imputed successfully")
            return df_imputed
        except Exception as e:
            raise CustomException(e, sys)
    
    def validate_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            
            expected_columns = list(self.schema.keys())
            target_info = self.schema.get('TARGET_COLUMN', {})
            target_column = target_info.get('name', 'Price')
            status = True
            schema_dtypes = self.schema.get('COLUMNS_TYPE', {})
            if target_column not in expected_columns:
                expected_columns.append(target_column)
            #check no. of columns
            if len(dataframe.columns) != len(expected_columns):
                logging.error(f"column count mismatch. Expected: {len(expected_columns)}")
                return False
            
            #check for same columns present of different columns
            if set(dataframe.columns) != set(expected_columns):
                missing_cols = set(expected_columns) -set(dataframe.columns)
                extra_cols = set(dataframe.columns) - set(expected_columns)
                logging.error(f"column name mismatch. Missing : {missing_cols}, Extra: {extra_cols}")
                return False
            #check for datatypes
            for col, expected_dtype in schema_dtypes.items():
                if col in dataframe.columns:
                    actual_dtype = str(dataframe[col].dtype)
                    if actual_dtype != expected_dtype:
                        logging.error(f"Data type mismatch for column: {col}. "
                                      f"Expected: {expected_dtype}, Found: {actual_dtype}")
                        return False
            logging.info("schema validation successful: all columns match")
            return True
    
        except Exception as e:
            raise CustomException(e, sys)
        
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            
            # Select only categorical columns for Chi-Square
            cat_cols = base_df.select_dtypes(include=['object']).columns
            
            for column in cat_cols:
                # 1. Get frequencies for both datasets
                base_counts = base_df[column].value_counts()
                current_counts = current_df[column].value_counts()
                
                # 2. Align the data (ensure both datasets have the same categories)
                df_counts = pd.DataFrame({
                    'base': base_counts,
                    'current': current_counts
                }).fillna(0) # Fill missing categories with 0
                
                # 3. Perform Chi-Square test on the aligned table
                chi2, p_value, dof, expected_freq = chi2_contingency(df_counts)
                
                is_found = p_value < threshold
                if is_found:
                    status = False
                
                report[column] = {
                    "p_value": float(p_value),
                    "drift_status": bool(is_found)
                }
                logging.info(f"Drift check for {column}: p_value={p_value:.4f}, drift={is_found}")
            
            # Save report
            write_yaml_file(file_path=str(self.data_validation_config.drift_report_file_name), content=report)
            return status
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_validation(self) -> DatavalidationArtifact:
        try:
            train_file_name = self.data_ingestion_artifact.training_file_name
            test_file_name = self.data_ingestion_artifact.testing_file_name

            # Read the data from train and test
            train_dataframe = DataValidation.read_data(train_file_name)
            test_dataframe = DataValidation.read_data(test_file_name)

            # Validate number of columns
            status = self.validate_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"train dataframe does not contain all columns.\n"
            status = self.validate_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"test dataframe does not contain all the columns.\n"
            
            # Impute null values using mode
            train_dataframe = self.impute_null_values(train_dataframe)
            test_dataframe = self.impute_null_values(test_dataframe)
            
            # Check dataset drift
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            
            # Create and save the data
            save_data(train_dataframe, self.data_validation_config.valid_train_file_name)
            save_data(test_dataframe, self.data_validation_config.valid_test_file_name)

            # Write validation status to STATUS_FILE
            status_report = {
                "validation_status": status,
                "message": "Data validation passed" if status else "Data validation failed - drift detected"
            }
            write_yaml_file(file_path=str(self.data_validation_config.Status_File), content=status_report)
            logging.info(f"Validation status saved to {self.data_validation_config.Status_File}")

            data_validation_artifact = DatavalidationArtifact(
                validation_status=Path(self.data_validation_config.Status_File),
                valid_train_file_name=Path(self.data_validation_config.valid_train_file_name),
                valid_test_file_name=Path(self.data_validation_config.valid_test_file_name),
                invalid_train_file_name=None,
                invalid_test_file_name=None,
                drift_report_file_name=Path(self.data_validation_config.drift_report_file_name))
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
