from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_file_name: Path
    testing_file_name: Path

@dataclass
class DatavalidationArtifact:
    validation_status: Path
    valid_train_file_name: Path
    valid_test_file_name: Path
    invalid_train_file_name: Path
    invalid_test_file_name: Path
    drift_report_file_name: Path