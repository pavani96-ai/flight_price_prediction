from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
import sys


class Flight_Price_Prediction_Model:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model

        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self,x):
        try: 
            x_transform = self.preprocessor.transform(x)
            y_hat =self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise CustomException(e, sys)
