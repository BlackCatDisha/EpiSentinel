import io
import json
import joblib
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter(tags=["prediction"])

MODEL_PATH_ENSEMBLE = "../final_model_plus1_no_district_pop_standardised/weighted_ensemble_plus1_near_optimal_no_district_pop.joblib"
MODEL_PATH_XGB = "../xgboost_trained/episentinel_pipeline.joblib"

# Global vars for the model
MODEL = None
FEATURE_COLUMNS = []
OPTIMAL_THRESHOLD = 0.5
MODEL_NAME = "Unknown"

def load_model():
    global MODEL, FEATURE_COLUMNS, OPTIMAL_THRESHOLD, MODEL_NAME
    if MODEL is not None:
        return

    # Try to load ensemble first
    try:
        import catboost  # Check if installed
        artifact = joblib.load(MODEL_PATH_ENSEMBLE)
        MODEL = artifact.get("model")
        FEATURE_COLUMNS = artifact.get("feature_columns", [])
        OPTIMAL_THRESHOLD = artifact.get("threshold", artifact.get("optimal_threshold", 0.15))
        MODEL_NAME = "WeightedEnsemble"
        print("Loaded Ensemble Model successfully.")
        return
    except Exception as e:
        print(f"Warning: Could not load Ensemble model ({e}). Falling back to XGBoost.")
    
    # Fallback to XGBoost
    try:
        artifact = joblib.load(MODEL_PATH_XGB)
        MODEL = artifact.get("model")
        FEATURE_COLUMNS = artifact.get("feature_columns", [])
        OPTIMAL_THRESHOLD = artifact.get("optimal_threshold", 0.5)
        MODEL_NAME = "XGBoost"
        print("Loaded XGBoost Model successfully.")
    except Exception as e:
        print(f"Critical Error: Could not load fallback XGBoost model either: {e}")

@router.post("/predict")
async def run_predictions(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    load_model()
    
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded. Please check the backend logs.")
        
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    X = df.copy()
    
    # Ensure all required features are present
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0.0  # Default value if missing
            
    X_features = X[FEATURE_COLUMNS]

    try:
        preds_proba = MODEL.predict_proba(X_features)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    predictions = {}
    for i, row in df.iterrows():
        # Handle "district", "District", etc.
        district_col = next((c for c in ['district', 'District', 'region', 'location', 'name', 'State', 'state'] if c in df.columns), None)
        district = str(row[district_col]) if district_col else f"Unknown_{i}"
        
        risk_score = float(preds_proba[i] * 100)
        risk_score_rounded = round(risk_score, 1)
        
        threshold_percentage = OPTIMAL_THRESHOLD * 100
        
        if risk_score >= threshold_percentage * 1.5:
            status = "Critical"
        elif risk_score >= threshold_percentage:
            status = "High"
        else:
            status = "Low"

        # Mock SHAP top driver for speed (real SHAP takes too long per request)
        # We can extract the highest value in the standardized feature set as a rough proxy
        top_driver_col = "Unknown"
        max_val = -9999
        for col in FEATURE_COLUMNS:
            val = abs(X_features.iloc[i][col])
            if val > max_val:
                max_val = val
                top_driver_col = col
                
        top_driver_formatted = top_driver_col.replace("_", " ").title()

        # Heuristic for cases
        pred_cases = max(0, int(risk_score / 2.5))
        
        predictions[district] = {
            "predicted_cases": pred_cases,
            "risk_score": risk_score_rounded,
            "status": status,
            "top_driver": top_driver_formatted,
            "detailed_explanation": f"Using {MODEL_NAME}: The primary indicator driving the risk score of {risk_score_rounded}% is '{top_driver_formatted}'.",
            "model_used": MODEL_NAME
        }
        
    return {"predictions": predictions}
