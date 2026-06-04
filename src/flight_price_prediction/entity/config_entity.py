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