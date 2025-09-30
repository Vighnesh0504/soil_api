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
    
class fertilizerFeatures(BaseModel):
    Temparature: float
    Humidity: float
    Moisture: float
    Soil_Type: object
    Crop_Type: object
    Nitrogen: float
    Potassium: float
    Phosphorous: float
    
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

soil_map={'Black':	0,'Clayey':	1,'Loamy':	2,'Red':	3,'Sandy':	4}
crop_map={'Barley':	0,'Cotton':	1,'Ground Nuts':	2,'Maize':	3,'Millets':	4,'Oil seeds'	:5,'Paddy':	6,'Pulses':	7,'Sugarcane'	:8,
'Tobacco':	9,
'Wheat':	10,
'coffee':	11,
'kidneybeans':	12,
'orange':	13,
'pomegranate':	14,
'rice':	15,
'watermelon':	16
}

fertilizer_map = {
    "10-10-10": 0,
    "10-26-26": 1,
    "14-14-14": 2,
    "14-35-14": 3,
    "15-15-15": 4,
    "17-17-17": 5,
    "20-20": 6,
    "28-28": 7,
    "DAP": 8,
    "Potassium chloride": 9,
    "Potassium sulfate.": 10,
    "Superphosphate": 11,
    "TSP": 12,
    "Urea": 13
}



# Reverse dictionary for decoding
decode_map = {v: k for k, v in encoding.items()}
fertlizerdecode_map = {v: k for k, v in fertilizer_map.items()}

app = FastAPI(title="Soil Fertility Prediction API")

# Load trained model
model = joblib.load("soil_fertility_model.pkl")
model1=joblib.load("crop_recomander_model.pkl")
model2=joblib.load("fertilizer_recomandation.pkl")



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
def predict(data: cropFeatures, top_n: int = 3):
    # Convert input to numpy array
    features = [[
        data.N, data.P, data.K, data.temperature, data.humidity,
        data.ph, data.rainfall, data.soil_health
    ]]

    # Predict
    probs = model1.predict_proba(features)[0]  # shape (num_classes,)

    # Get indices of top_n highest probabilities
    top_indices = np.argsort(probs)[::-1][:top_n]

    # Map indices back to crop names
    recommendations = [
        {
            "crop": decode_map[int(idx)],
            "probability": float(probs[idx])
        }
        for idx in top_indices
    ]

    return {"recommendations": recommendations}

@app.post("/predict/fertilizer")
def predict(data: fertilizerFeatures):
    
    soil_type_encoded = soil_map[data.Soil_Type]
    crop_type_encoded = crop_map[data.Crop_Type]
    
    features = [[
        data.Temparature, data.Humidity, data.Moisture, soil_type_encoded, crop_type_encoded,
        data.Nitrogen, data.Potassium, data.Phosphorous
    ]]
    
    prediction = model2.predict(features)
    
    recommandation=fertlizerdecode_map[prediction[0]]
    
    return {'fertilizer':recommandation}
    
    


