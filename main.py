# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# FastAPI app
app = FastAPI(title="Iris Prediction API")

# Input format
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
