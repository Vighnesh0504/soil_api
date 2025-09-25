from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Define your input schema (match dataset columns)
class SoilFeatures(BaseModel):
    N: float
    P: float
    K: float
    pH: float
    EC: float
    OC: float
    S: float
    Zn: float
    Fe: float
    Cu: float
    Mn: float
    B: float

# Create FastAPI app
app = FastAPI(title="Soil Fertility Prediction API")

# Load trained model
model = joblib.load("soil_fertility_model.pkl")  # <-- replace with your filename

@app.post("/predict")
def predict(data: SoilFeatures):
    # Convert input to numpy array
    features = [[
        data.N, data.P, data.K, data.pH, data.EC,
        data.OC, data.S, data.Zn, data.Fe,
        data.Cu, data.Mn, data.B
    ]]

    # Predict
    prediction = model.predict(features)

    if prediction[0] == 0:
        result = "low-compost"
    elif prediction[0] == 1:
        result = "medium-compost"
    else:
        result = "high-compost"

    return {"prediction": result}
