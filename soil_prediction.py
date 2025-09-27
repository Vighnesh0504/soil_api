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
    
class cropFeatures(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    soil_health: float
    
# Create FastAPI app

encoding = {
    'apple': 0, 'banana': 1, 'blackgram': 2,
    'chickpea': 3, 'coconut': 4, 'coffee': 5,
    'cotton': 6, 'grapes': 7, 'jute': 8,
    'kidneybeans': 9, 'lentil': 10, 'maize': 11,
    'mango': 12, 'mothbeans': 13, 'mungbean': 14,
    'muskmelon': 15, 'orange': 16, 'papaya': 17,
    'pigeonpeas': 18, 'pomegranate': 19, 'rice': 20,
    'watermelon': 21
}

# Reverse dictionary for decoding
decode_map = {v: k for k, v in encoding.items()}

app = FastAPI(title="Soil Fertility Prediction API")

# Load trained model
model = joblib.load("soil_fertility_model.pkl")
model1=joblib.load("crop_recomander_model.pkl")

@app.post("/predict/soil")
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

@app.post("/predict/crop")
def predict(data: cropFeatures):
    # Convert input to numpy array
    features = [[
        data.N, data.P, data.K, data.temperature, data.humidity,
        data.ph, data.rainfall, data.soil_health
    ]]

    # Predict
    prediction = model.predict(features)

    crop_name = decode_map[int(prediction[0])]

    return {"prediction": crop_name}


