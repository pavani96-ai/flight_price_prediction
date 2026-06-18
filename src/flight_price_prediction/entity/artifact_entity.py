from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_file_name: Path
    testing_file_name: Path

@dataclass
class DataValidationArtifact:
    validation_status: Path
    valid_train_file_name: Path
    valid_test_file_name: Path
    invalid_train_file_name: Path
    invalid_test_file_name: Path
    drift_report_file_name: Path

@dataclass
class FeatureEngineeringArtifact:
    engineered_train_file_name: Path
    engineered_test_file_name: Path

@dataclass
class DataTransformationArtifact:
    transformed_train_file_name: Path
    transformed_test_file_name: Path
    preprocessor_object_file_name: Path

@dataclass
class RegressionMetricArtifact:
    r2_score: float
    adj_r2_score: float
    mean_absolute_error: float
    mean_squared_error: float


@dataclass
class ModelTrainerArtifact:
    model_file_name: Path
    train_metric_artifact: RegressionMetricArtifact
    test_metric_artifact: RegressionMetricArtifact
    
