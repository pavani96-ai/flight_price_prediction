from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    feature_store_file_name: Path
    training_file_name: Path
    testing_file_name: Path
    train_test_split_ratio: float
    collection_name: str
    database_name: str

@dataclass
class DataValidationConfig:
    root_dir: Path
    Status_File: Path
    validated_dir: Path
    invalid_dir: Path
    drift_report_dir: Path
    drift_report_file_name: Path
    valid_train_file_name: Path
    valid_test_file_name: Path
    all_schema: dict

@dataclass
class FeatureEngineeringConfig:
    root_dir: Path
    engineered_train_file_name: Path
    engineered_test_file_name: Path
    feature_engineered_pkl_file: Path
    target_column: str
    stop_map: dict
    replace_destinations: dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    transformed_train_file_name: Path
    transformed_test_file_name: Path
    preprocessor_object_file_name: Path
    target_column: str
    numerical_columns: list
    categorical_columns: list
    scaler: str
    encoder: str

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    trained_models_dir: Path
    model_file_name: Path
    expected_score: float
    over_fitting_underfitting_threshold: float

@dataclass
class S3SyncConfig:
    training_bucket_name: str
    artifact_dir: Path
    final_model_dir: Path