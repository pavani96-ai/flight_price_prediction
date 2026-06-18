import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from src.flight_price_prediction.entity.config_entity import FeatureEngineeringConfig
def _get_column_values(X, index):
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, index].to_numpy()
    return np.asarray(X)[:, index]

def _get_two_columns(X):
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0].to_numpy(), X.iloc[:, 1].to_numpy()
    arr = np.asarray(X)
    return arr[:, 0], arr[:, 1]

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        values = _get_column_values(X, 0)
        dates = pd.to_datetime(values, dayfirst=True, errors='coerce')
        if hasattr(dates, 'dt'):
            month = dates.dt.month.fillna(0).astype(int).to_numpy().reshape(-1, 1)
            day = dates.dt.weekday.fillna(0).astype(int).to_numpy().reshape(-1, 1)
        else:
            month = pd.Series(dates.month).fillna(0).astype(int).to_numpy().reshape(-1, 1)
            day = pd.Series(dates.weekday).fillna(0).astype(int).to_numpy().reshape(-1, 1)

        return np.hstack([month, day])
    
class DurationMinutesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 

    @staticmethod
    def _to_minutes(duration):
        if not isinstance(duration, str):
            return 0
        duration = duration.strip()
        if 'h' in duration and 'm' in duration:
            hours = int(duration.split('h')[0].strip())
            minutes = int(duration.split('h')[1].replace('m', '').strip())
            return hours * 60 + minutes
        if 'h' in duration:
            return int(duration.replace('h', '').strip()) * 60
        if 'm' in duration:
            return int(duration.replace('m', '').strip())
        return 0
    
    def transform(self, X):
        values = [self._to_minutes(value) for value in _get_column_values(X, 0)]
        return np.array(values, dtype=int).reshape(-1, 1)
    
class TimeCyclicTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    @staticmethod
    def _to_minutes(time_str):
       try:
           parsed = pd.to_datetime(time_str, format='%H:%M', errors='coerce')
           if pd.isna(parsed):
               return 0
           return int(parsed.hour) * 60 + int(parsed.minute)
       except Exception:
           return 0

    def transform(self, X):
        values = [self._to_minutes(value) for value in _get_column_values(X, 0)]
        minutes = np.array(values, dtype=float)
        sin_values = np.sin(2 * np.pi * minutes / 1440)
        cos_values = np.cos(2 * np.pi * minutes / 1440)
        return np.vstack([sin_values, cos_values]).T

class TotalStopsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_map):
        self.stop_map =stop_map 

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        values =[self.stop_map.get(str(value).strip(), 0) for value in _get_column_values(X, 0)]
        return np.array(values, dtype=int).reshape(-1,1)
    
class DestinationNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, replace_destinations):
        self.replace_destinations = replace_destinations

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        normalized = [self.replace_destinations.get(str(value).strip(), str(value).strip()) for value in _get_column_values(X, 0)]
        return np.array(normalized, dtype=object).reshape(-1, 1)
    

class RouteInteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, replace_destinations):
        self.replace_destinations = replace_destinations

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        source, destination = _get_two_columns(X)
        source = [str(val).strip() for val in source]
        destination = [self.replace_destinations.get(str(val).strip(), str(val).strip()) for val in destination]
        routes = [f"{src}_{dest}" for src, dest in zip(source, destination)]
        return np.array(routes, dtype=object).reshape(-1, 1)
    
class DurationStopsInteractionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    @staticmethod
    def _to_minutes(duration):
        if not isinstance(duration, str):
            return 0.0
        duration = duration.strip()
        if 'h' in duration and 'm' in duration:
            hours = int(duration.split('h')[0].strip())
            minutes = int(duration.split('h')[1].replace('m', '').strip())
            return float(hours * 60 + minutes)
        if 'h' in duration:
            return float(int(duration.replace('h', '').strip()) * 60)
        if 'm' in duration:
            return float(int(duration.replace('m', '').strip()))
        return 0.0

    def transform(self, X):
        # X will be a DataFrame or array containing [Duration, Total_Stops]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['duration', 'total_stops'])

        duration_values = [self._to_minutes(val) for val in X.iloc[:, 0].to_numpy()]
        stop_values = []
        for val in X.iloc[:, 1].to_numpy():
            try:
                stop_values.append(float(str(val).strip()))
            except ValueError:
                stop_values.append(0.0)

        duration = np.array(duration_values, dtype=float)
        stops = np.array(stop_values, dtype=float)

        interaction = (duration * stops).reshape(-1, 1)
        return interaction


