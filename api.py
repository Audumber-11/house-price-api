from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import os


app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting California house prices",
    version="1.0.0"
)


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")


class PredictionResponse(BaseModel):
    predicted_price: float
    price_in_dollars: float
    input_features: HouseFeatures


def load_models():
    model_path = os.path.join(os.path.dirname(__file__), 'models')
    try:
        scaler = joblib.load(f'{model_path}/scaler.pkl')
        model = joblib.load(f'{model_path}/model.pkl')
        feature_names = joblib.load(f'{model_path}/feature_names.pkl')
        return scaler, model, feature_names
    except FileNotFoundError:
        return None, None, None


def create_features(df: HouseFeatures):
    MedInc = df.MedInc
    HouseAge = df.HouseAge
    AveRooms = df.AveRooms
    AveBedrms = df.AveBedrms
    Population = df.Population
    AveOccup = df.AveOccup
    Latitude = df.Latitude
    Longitude = df.Longitude
    
    RoomsPerHousehold = AveRooms / AveOccup
    BedroomsRatio = AveBedrms / AveRooms
    PopulationPerOccup = Population / AveOccup
    
    MedInc_squared = MedInc ** 2
    HouseAge_squared = HouseAge ** 2
    
    MedInc_HouseAge = MedInc * HouseAge
    MedInc_AveRooms = MedInc * AveRooms
    
    Location_Score = abs(40 - Latitude) + abs(-120 - Longitude)
    
    return np.array([[
        MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude,
        RoomsPerHousehold, BedroomsRatio, PopulationPerOccup,
        MedInc_squared, HouseAge_squared, MedInc_HouseAge, MedInc_AveRooms, Location_Score
    ]])


scaler, model, feature_names = load_models()


@app.get("/")
def root():
    return {"message": "House Price Prediction API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")
    
    feature_array = create_features(features)
    feature_scaled = scaler.transform(feature_array)
    prediction = model.predict(feature_scaled)[0]
    
    return PredictionResponse(
        predicted_price=round(prediction, 4),
        price_in_dollars=round(prediction * 100000, 2),
        input_features=features
    )


@app.post("/predict_batch")
def predict_batch(features_list: List[HouseFeatures]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for features in features_list:
        feature_array = create_features(features)
        feature_scaled = scaler.transform(feature_array)
        pred = model.predict(feature_scaled)[0]
        predictions.append({
            "predicted_price": round(pred, 4),
            "price_in_dollars": round(pred * 100000, 2)
        })
    
    return {"predictions": predictions, "count": len(predictions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)