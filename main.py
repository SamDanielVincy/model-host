from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np

# Create app
app = FastAPI(title="ML Model API", description="A simple ML model deployed on Cloud Run", version="1.0")

# Input schema
class InputData(BaseModel):
    features: list

# Load trained model (saved earlier as model.pkl)
model = joblib.load("model.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to ML Model API on Cloud Run!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        X = np.array(data.features).reshape(1, -1)
        prediction = model.predict(X)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}

# Run locally (for testing)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
