from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import pickle
import os

# =========================
# INITIALIZE FASTAPI APP
# =========================
app = FastAPI(
    title="Flat Price Prediction API",
    description="API for predicting flat prices using CatBoost",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# LOAD MODEL AND METADATA
# =========================
MODEL_PATH = "models/flat_price_prediction_model.cbm"
CAT_COLS_PATH = "models/cat_cols.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

# Check if model files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")

# Load model
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# Load categorical columns
with open(CAT_COLS_PATH, "rb") as f:
    cat_cols = pickle.load(f)

# Load feature names
with open(FEATURE_NAMES_PATH, "rb") as f:
    feature_names = pickle.load(f)

print(f"✅ Model loaded successfully")
print(f"✅ Categorical columns: {cat_cols}")
print(f"✅ Total features: {len(feature_names)}")

# =========================
# PYDANTIC MODELS
# =========================
class FlatInput(BaseModel):
    """Input model for a flat"""
    kitchen_area: int = Field(..., description="Kitchen area in square meters")
    bath_area: int = Field(..., description="Bathroom area in square meters")
    other_area: float = Field(..., description="Other area in square meters")
    gas: str = Field(..., description="Gas availability (e.g., 'yes', 'no')")
    hot_water: str = Field(..., description="Hot water availability")
    central_heating: str = Field(..., description="Central heating availability")
    extra_area: int = Field(..., description="Extra area in square meters")
    extra_area_count: int = Field(..., description="Count of extra areas")
    year: int = Field(..., description="Year of construction")
    ceil_height: float = Field(..., description="Ceiling height in meters")
    floor_max: int = Field(..., description="Maximum floor in building")
    floor: int = Field(..., description="Floor number")
    total_area: float = Field(..., description="Total area in square meters")
    bath_count: int = Field(..., description="Number of bathrooms")
    extra_area_type_name: str = Field(..., description="Type of extra area")
    district_name: str = Field(..., description="District name")
    rooms_count: int = Field(..., description="Number of rooms")

    class Config:
        json_schema_extra = {
            "example": {
                "kitchen_area": 10,
                "bath_area": 5,
                "other_area": 15.5,
                "gas": "yes",
                "hot_water": "yes",
                "central_heating": "yes",
                "extra_area": 20,
                "extra_area_count": 1,
                "year": 2010,
                "ceil_height": 2.7,
                "floor_max": 10,
                "floor": 5,
                "total_area": 65.5,
                "bath_count": 1,
                "extra_area_type_name": "balcony",
                "district_name": "Central",
                "rooms_count": 2
            }
        }


class FlatOutput(BaseModel):
    """Output model for prediction"""
    predicted_price: float = Field(..., description="Predicted price in currency units")
    log_price: float = Field(..., description="Log-transformed prediction")


# =========================
# HELPER FUNCTIONS
# =========================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the input dataframe
    """
    df = df.copy()
    
    # Create engineered features
    df['room_size'] = df['total_area'] / (df['rooms_count'] + 1)
    df['floor_ratio'] = df['floor'] / (df['floor_max'] + 1)
    df['is_top_floor'] = (df['floor'] == df['floor_max']).astype(int)
    df['is_first_floor'] = (df['floor'] == 1).astype(int)
    df['area_per_bath'] = df['total_area'] / (df['bath_count'] + 1)
    df['building_age'] = 2026 - df['year']
    
    # Convert categorical columns
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Ensure all expected features are present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the features used during training
    df = df[feature_names]
    
    return df


def predict_price(input_data: FlatInput) -> FlatOutput:
    """
    Make price prediction for a flat
    """
    # Convert input to dataframe
    input_dict = input_data.model_dump()
    df = pd.DataFrame([input_dict])
    
    # Engineer features
    df_processed = engineer_features(df)
    
    # Make prediction (in log space)
    log_pred = model.predict(df_processed)[0]
    
    # Transform back to original scale
    price_pred = np.expm1(log_pred)
    
    return FlatOutput(
        predicted_price=float(price_pred),
        log_price=float(log_pred)
    )


# =========================
# API ENDPOINTS
# =========================
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Flat Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict price for a flat",
            "GET /health": "Health check endpoint",
            "GET /model/info": "Get model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH
    }


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_type": "CatBoostRegressor",
        "total_features": len(feature_names),
        "categorical_features": cat_cols,
        "feature_names": feature_names,
        "model_params": {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 8,
            "loss_function": "RMSE"
        }
    }


@app.post("/predict", response_model=FlatOutput)
async def predict(flat: FlatInput):
    """
    Predict the price for a flat
        
    Returns the predicted price and log-transformed price
    """
    try:
        result = predict_price(flat)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("Starting Flat Price Prediction API")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"Features: {len(feature_names)}")
    print(f"Categorical columns: {len(cat_cols)}")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
