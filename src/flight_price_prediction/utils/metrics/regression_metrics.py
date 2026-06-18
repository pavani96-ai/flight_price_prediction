import sys
from src.flight_price_prediction.entity.artifact_entity import RegressionMetricArtifact
from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def get_regression_score(y_true, y_pred, n_features: int) -> RegressionMetricArtifact:
    try:
        model_r2_score = r2_score(y_true, y_pred)
        model_mean_absolute_error = mean_absolute_error(y_true, y_pred)
        model_mean_squared_error = mean_squared_error(y_true, y_pred)

        n_samples = len(y_true)
        adj_r2_score = model_r2_score
        if n_samples - n_features - 1 > 0:
            adj_r2_score = 1 - (1 - model_r2_score) * (n_samples - 1) / (n_samples - n_features - 1)

        regression_metric = RegressionMetricArtifact(
            r2_score = model_r2_score,
            adj_r2_score = adj_r2_score,
            mean_absolute_error= model_mean_absolute_error,
            mean_squared_error=model_mean_squared_error)
        return regression_metric
    except Exception as e:
        raise CustomException(e, sys)