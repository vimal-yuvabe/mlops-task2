from fastapi import FastAPI
import xgboost as xgb
import pandas as pd
from pydantic import BaseModel

# Load pre-trained model
MODEL_PATH = "./model.bst"
model = xgb.Booster()
model.load_model(MODEL_PATH)

app = FastAPI()

class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    dmatrix = xgb.DMatrix([request.features])
    prediction = model.predict(dmatrix)
    return {"prediction": prediction.tolist()}
