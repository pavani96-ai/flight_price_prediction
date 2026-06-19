import sys

from dotenv import load_dotenv
load_dotenv()

from src.flight_price_prediction.exception.exception import CustomException
from src.flight_price_prediction.logging.logger import logging
from src.flight_price_prediction.pipeline.train_pipeline import TrainPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, Request, UploadFile
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.flight_price_prediction.utils.common import load_bin, create_directory
from src.flight_price_prediction.utils.ml_utils.model import Flight_Price_Prediction_Model

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins =origins,
    allow_credentials = True,
    allow_methods =['*'],
    allow_headers = ['*'],

)

templates = Jinja2Templates(directory ="./templates")

@app.get("/", tags=["home"])
async def home_route(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

@app.get("/train", tags=["training"])
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise CustomException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile= File(...)):
    try:
        df =pd.read_csv(file.file)
        feature_engineered = load_bin("final_model/feature_engineering.pkl")
        processed_data = feature_engineered.transform(df)
        #print(processed_data)

        preprocessor = load_bin("final_model/preprocessor.pkl")
        final_model = load_bin('final_model/model.pkl')
        flight_price_prediction_model =Flight_Price_Prediction_Model(preprocessor = preprocessor,model =final_model)
        print(df.iloc[0])
        y_pred = flight_price_prediction_model.predict(processed_data)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        create_directory("prediction_output/output.csv")
        df.to_csv("prediction_output/output.csv", index=False)
        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"table": table_html},
        )

    except Exception as e:
        raise CustomException(e, sys)

if __name__=="__main__":
    app_run(app, host= "0.0.0.0", port=8000)

# increased volume in ec2 instance
