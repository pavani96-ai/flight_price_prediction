from src.flight_price_prediction.constants import *
from src.flight_price_prediction.utils.common import read_yaml, create_directories
from src.flight_price_prediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    FeatureEngineeringConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    S3SyncConfig,
)
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class ConfigurationManager:
    def __init__(self,
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            feature_store_file_name=Path(config.feature_store_file_name),
            training_file_name=Path(config.training_file_name),
            testing_file_name=Path(config.testing_file_name),
            train_test_split_ratio = params.train_test_split_ratio,
            collection_name=config.collection_name,
            database_name=config.database_name
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
         config = self.config.data_validation
         schema = self.schema.COLUMNS

         data_validation_dir = Path(config.root_dir)
         validated_dir = Path(config.validated_dir)
         invalid_dir = Path(config.invalid_dir)
         drift_report_dir = Path(config.drift_report_dir)
         valid_train_file_name = Path(config.valid_train_file_name)
         valid_test_file_name = Path(config.valid_test_file_name)
         drift_report_file_name = Path(config.drift_report_file_name)
         Status_File = Path(config.Status_File)


         create_directories([data_validation_dir,validated_dir,invalid_dir,
                             drift_report_dir])
         data_validation_config = DataValidationConfig(
             root_dir = data_validation_dir,
             Status_File = Status_File,
             validated_dir = validated_dir,
             invalid_dir = invalid_dir,
             drift_report_dir = drift_report_dir,
             valid_train_file_name = valid_train_file_name,
             valid_test_file_name = valid_test_file_name,
             drift_report_file_name = drift_report_file_name,
             all_schema = schema,
           )
         return data_validation_config

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        config = self.config.feature_engineering
        params = self.params.feature_engineering

        engineered_train_file_name = Path(config.engineered_train_file_name)
        engineered_test_file_name = Path(config.engineered_test_file_name)
        feature_engineered_pkl_file = Path(config.feature_engineered_pkl_file)
        
        create_directories([
            Path(config.root_dir),
            engineered_train_file_name.parent,
            engineered_test_file_name.parent,
            feature_engineered_pkl_file.parent
        ])

        return FeatureEngineeringConfig(
            root_dir=Path(config.root_dir),
            engineered_train_file_name=engineered_train_file_name,
            engineered_test_file_name=engineered_test_file_name,
            feature_engineered_pkl_file = feature_engineered_pkl_file,
            target_column=params.target_column,
            stop_map=dict(params.stop_map),
            replace_destinations=dict(params.replace_destinations),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation

        transformed_train_file_name = Path(config.transformed_train_file_name)
        transformed_test_file_name = Path(config.transformed_test_file_name)
        preprocessor_object_file_name = Path(config.preprocessor_object_file_name)
        target_column=params.target_column
        numerical_columns=list(params.numerical_columns)
        categorical_columns=list(params.categorical_columns)
        scaler=StandardScaler()
        encoder=OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        

        create_directories([
            Path(config.root_dir),
            transformed_train_file_name.parent,
            transformed_test_file_name.parent,
            preprocessor_object_file_name.parent,
        ])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            transformed_train_file_name=transformed_train_file_name,
            transformed_test_file_name=transformed_test_file_name,
            preprocessor_object_file_name=preprocessor_object_file_name,
            target_column=target_column,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            scaler=StandardScaler(),
            encoder=OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    

    def get_model_training_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        model_trainer_dir = Path(config.root_dir)
        trained_models_dir = Path(config.trained_models_dir)
        model_file_name = Path(config.model_file_name)
        expected_score = params.expected_score
        over_fitting_underfitting_threshold = params.over_fitting_underfitting_threshold

        create_directories([model_trainer_dir,trained_models_dir.parent,model_file_name.parent])

        return ModelTrainerConfig(root_dir = model_trainer_dir,
        trained_models_dir = trained_models_dir,
        model_file_name = model_file_name ,
        expected_score = expected_score,
        over_fitting_underfitting_threshold = over_fitting_underfitting_threshold

        )

    def get_s3_sync_config(self) -> S3SyncConfig:
        return S3SyncConfig(
            training_bucket_name=self.config.Training_Bucket_Name,
            artifact_dir=Path(self.config.artifacts_root),
            final_model_dir=Path(getattr(self.config, "final_model_dir", "final_model")),
        )

    