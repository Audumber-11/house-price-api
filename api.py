from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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


UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <style>
        *, *::before, *::after {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 720px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
            animation: fadeIn 0.8s ease-out;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.5px;
        }

        .header p {
            color: #a0a0b8;
            margin-top: 0.5rem;
            font-size: 0.95rem;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            animation: slideUp 0.6s ease-out;
        }

        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #c0c0e0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-title svg {
            width: 20px;
            height: 20px;
            fill: #667eea;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .form-group.full-width {
            grid-column: 1 / -1;
        }

        .form-group label {
            font-size: 0.8rem;
            font-weight: 500;
            color: #9090b0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .form-group label .hint {
            color: #606080;
            font-weight: 400;
            text-transform: none;
        }

        .form-group input {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 0.7rem 1rem;
            color: #e0e0e0;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            outline: none;
            width: 100%;
        }

        .form-group input:focus {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.08);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
        }

        .form-group input:hover {
            border-color: rgba(255, 255, 255, 0.2);
        }

        .form-group input::placeholder {
            color: #505070;
        }

        .form-group input.error {
            border-color: #ef4444;
            background: rgba(239, 68, 68, 0.08);
        }

        .form-group .error-msg {
            font-size: 0.75rem;
            color: #ef4444;
            display: none;
        }

        .form-group .error-msg.visible {
            display: block;
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 0.5rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 12px;
            padding: 0.9rem;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .submit-btn:active:not(:disabled) {
            transform: translateY(0);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .submit-btn .spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }

        .submit-btn.loading .spinner {
            display: inline-block;
        }

        .submit-btn.loading .btn-text {
            display: none;
        }

        .result-card {
            display: none;
            margin-top: 1.5rem;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            animation: slideUp 0.4s ease-out;
        }

        .result-card.visible {
            display: block;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .result-label {
            font-size: 0.85rem;
            color: #9090b0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .result-price {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #4ade80, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .result-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .detail-item {
            display: flex;
            justify-content: space-between;
            padding: 0.35rem 0;
            font-size: 0.85rem;
        }

        .detail-item .key {
            color: #707090;
        }

        .detail-item .value {
            color: #c0c0e0;
            font-weight: 500;
        }

        .error-banner {
            display: none;
            margin-top: 1rem;
            padding: 0.8rem 1rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: 10px;
            color: #fca5a5;
            font-size: 0.9rem;
            animation: shake 0.4s ease-out;
        }

        .error-banner.visible {
            display: block;
        }

        .footer {
            text-align: center;
            margin-top: 1.5rem;
            color: #505070;
            font-size: 0.8rem;
        }

        .footer a {
            color: #667eea;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }

        @media (max-width: 600px) {
            body { padding: 1rem; }
            .card { padding: 1.25rem; }
            .form-grid { grid-template-columns: 1fr; }
            .result-details { grid-template-columns: 1fr; }
            .header h1 { font-size: 1.5rem; }
            .result-price { font-size: 1.6rem; }
        }

        .batch-section {
            margin-top: 1rem;
            text-align: center;
        }

        .batch-btn {
            background: transparent;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 0.5rem 1rem;
            color: #9090b0;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .batch-btn:hover {
            border-color: #667eea;
            color: #c0c0e0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 House Price Predictor</h1>
            <p>Enter property details to get an estimated market value using our ML model</p>
        </div>

        <div class="card">
            <div class="card-title">
                <svg viewBox="0 0 24 24" width="20" height="20"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>
                Property Features
            </div>
            <form id="predictForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="MedInc">MedInc <span class="hint">(median income)</span></label>
                        <input type="number" id="MedInc" name="MedInc" value="5.0" step="0.1" min="0" max="20" placeholder="e.g. 5.0">
                    </div>
                    <div class="form-group">
                        <label for="HouseAge">HouseAge <span class="hint">(years)</span></label>
                        <input type="number" id="HouseAge" name="HouseAge" value="20.0" step="1" min="0" max="100" placeholder="e.g. 20">
                    </div>
                    <div class="form-group">
                        <label for="AveRooms">AveRooms <span class="hint">(per household)</span></label>
                        <input type="number" id="AveRooms" name="AveRooms" value="6.0" step="0.1" min="0" max="50" placeholder="e.g. 6.0">
                    </div>
                    <div class="form-group">
                        <label for="AveBedrms">AveBedrms <span class="hint">(per household)</span></label>
                        <input type="number" id="AveBedrms" name="AveBedrms" value="1.0" step="0.1" min="0" max="10" placeholder="e.g. 1.0">
                    </div>
                    <div class="form-group">
                        <label for="Population">Population <span class="hint">(block group)</span></label>
                        <input type="number" id="Population" name="Population" value="1000.0" step="10" min="0" max="50000" placeholder="e.g. 1000">
                    </div>
                    <div class="form-group">
                        <label for="AveOccup">AveOccup <span class="hint">(household size)</span></label>
                        <input type="number" id="AveOccup" name="AveOccup" value="3.0" step="0.1" min="0" max="50" placeholder="e.g. 3.0">
                    </div>
                    <div class="form-group">
                        <label for="Latitude">Latitude</label>
                        <input type="number" id="Latitude" name="Latitude" value="34.0" step="0.01" min="32" max="42" placeholder="e.g. 34.0">
                    </div>
                    <div class="form-group">
                        <label for="Longitude">Longitude</label>
                        <input type="number" id="Longitude" name="Longitude" value="-118.0" step="0.01" min="-125" max="-114" placeholder="e.g. -118.0">
                    </div>
                    <button type="submit" class="submit-btn" id="submitBtn">
                        <span class="spinner"></span>
                        <span class="btn-text">🔮 Predict Price</span>
                    </button>
                </div>
            </form>
        </div>

        <div class="result-card" id="resultCard">
            <div class="result-header">
                <span class="result-label">Estimated Market Value</span>
            </div>
            <div class="result-price" id="resultPrice">$0</div>
            <div class="result-details" id="resultDetails"></div>
        </div>

        <div class="error-banner" id="errorBanner"></div>

        <div class="batch-section">
            <button class="batch-btn" onclick="loadSampleData()">📋 Load Sample Data</button>
            <span style="color: #404060; margin: 0 0.5rem;">|</span>
            <button class="batch-btn" onclick="resetForm()">🔄 Reset</button>
        </div>

        <div class="footer">
            Built with scikit-learn &middot; XGBoost model &middot; R² = 0.84
        </div>
    </div>

    <script>
        const form = document.getElementById('predictForm');
        const submitBtn = document.getElementById('submitBtn');
        const resultCard = document.getElementById('resultCard');
        const resultPrice = document.getElementById('resultPrice');
        const resultDetails = document.getElementById('resultDetails');
        const errorBanner = document.getElementById('errorBanner');

        function getFormData() {
            return {
                MedInc: parseFloat(document.getElementById('MedInc').value) || 0,
                HouseAge: parseFloat(document.getElementById('HouseAge').value) || 0,
                AveRooms: parseFloat(document.getElementById('AveRooms').value) || 0,
                AveBedrms: parseFloat(document.getElementById('AveBedrms').value) || 0,
                Population: parseFloat(document.getElementById('Population').value) || 0,
                AveOccup: parseFloat(document.getElementById('AveOccup').value) || 0,
                Latitude: parseFloat(document.getElementById('Latitude').value) || 0,
                Longitude: parseFloat(document.getElementById('Longitude').value) || 0
            };
        }

        function loadSampleData() {
            const samples = [
                { MedInc: 5.0, HouseAge: 20, AveRooms: 6.0, AveBedrms: 1.0, Population: 1000, AveOccup: 3.0, Latitude: 34.0, Longitude: -118.0 },
                { MedInc: 8.3, HouseAge: 15, AveRooms: 7.2, AveBedrms: 1.1, Population: 1200, AveOccup: 2.8, Latitude: 37.8, Longitude: -122.2 },
                { MedInc: 2.1, HouseAge: 45, AveRooms: 4.5, AveBedrms: 1.3, Population: 2500, AveOccup: 3.5, Latitude: 33.9, Longitude: -118.2 },
            ];
            const sample = samples[Math.floor(Math.random() * samples.length)];
            for (const [key, value] of Object.entries(sample)) {
                document.getElementById(key).value = value;
            }
            resultCard.classList.remove('visible');
            errorBanner.classList.remove('visible');
        }

        function resetForm() {
            form.reset();
            resultCard.classList.remove('visible');
            errorBanner.classList.remove('visible');
        }

        function formatDollars(amount) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            }).format(amount);
        }

        function formatNumber(value, decimals) {
            return new Intl.NumberFormat('en-US', {
                minimumFractionDigits: decimals || 0,
                maximumFractionDigits: decimals || 2
            }).format(value);
        }

        async function handleSubmit(e) {
            e.preventDefault();
            
            errorBanner.classList.remove('visible');
            submitBtn.classList.add('loading');
            submitBtn.disabled = true;

            const data = getFormData();

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Prediction failed');
                }

                const result = await response.json();

                resultCard.classList.remove('visible');
                
                setTimeout(() => {
                    resultPrice.textContent = formatDollars(result.price_in_dollars);
                    
                    const details = [
                        { key: 'Raw Score', value: result.predicted_price.toFixed(4) },
                        { key: 'Median Income', value: formatNumber(data.MedInc, 1) },
                        { key: 'House Age', value: formatNumber(data.HouseAge, 0) + ' yrs' },
                        { key: 'Avg Rooms', value: formatNumber(data.AveRooms, 1) },
                        { key: 'Avg Bedrooms', value: formatNumber(data.AveBedrms, 1) },
                        { key: 'Population', value: formatNumber(data.Population, 0) },
                        { key: 'Avg Occupancy', value: formatNumber(data.AveOccup, 1) },
                        { key: 'Latitude', value: formatNumber(data.Latitude, 2) },
                        { key: 'Longitude', value: formatNumber(data.Longitude, 2) },
                    ];

                    resultDetails.innerHTML = details.map(d => 
                        `<div class="detail-item"><span class="key">${d.key}</span><span class="value">${d.value}</span></div>`
                    ).join('');

                    resultCard.classList.add('visible');
                }, 100);

            } catch (err) {
                errorBanner.textContent = '⚠\uFE0F ' + err.message;
                errorBanner.classList.add('visible');
            } finally {
                submitBtn.classList.remove('loading');
                submitBtn.disabled = false;
            }
        }

        form.addEventListener('submit', handleSubmit);

        window.addEventListener('load', () => {
            loadSampleData();
        });
    </script>
</body>
</html>
"""

scaler, model, feature_names = load_models()


@app.get("/")
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=UI_HTML)
    return {"message": "House Price Prediction API", "version": "1.0.0", "ui": "Visit / in a browser for the web interface"}


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