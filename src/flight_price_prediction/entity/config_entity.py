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
