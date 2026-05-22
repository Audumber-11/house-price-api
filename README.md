# House Price Prediction Pipeline

End-to-end ML pipeline for predicting California house prices.

## Features

- Data cleaning with outlier removal
- Feature engineering (9 new features)
- EDA with visualizations
- Model training (XGBoost, Random Forest, Gradient Boosting)
- FastAPI deployment

## Quick Start

```bash
pip install -r requirements.txt
python model_training.py
uvicorn api:app --reload
```

## Web UI

Visit `http://localhost:8000/` in your browser for an interactive prediction form with:
- Dark-themed UI with glassmorphism design
- Pre-loaded sample data
- Real-time price prediction
- Async loading states and error handling

## API Endpoints

- `GET /` - Web UI (browser) or API info (JSON)
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

## Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 5.0,
    "HouseAge": 20.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
  }'
```

## Deployment

### Render (Recommended)

1. Push to GitHub
2. Connect to Render
3. Deploy from `render.yaml`

### Docker

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

## Project Structure

```
house-price-prediction/
├── data_cleaning.py    # Data preprocessing
├── eda.py              # Exploratory analysis
├── model_training.py   # Model training pipeline
├── api.py              # FastAPI service
├── requirements.txt    # Dependencies
├── Dockerfile          # Container config
└── render.yaml         # Render deployment
```