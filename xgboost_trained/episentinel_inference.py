"""
EpiSentinel — Inference Script
Load the saved pipeline and predict outbreak probability for new data.
"""

import json
import numpy as np
import pandas as pd
import joblib

# ── 1. Load artifacts ──────────────────────────────────────────────────────
artifact        = joblib.load("episentinel_pipeline.joblib")
model           = artifact["model"]
le              = artifact["label_encoder"]        # None if native categorical used
use_native_cat  = artifact["use_native_cat"]
feature_columns = artifact["feature_columns"]
threshold       = artifact["optimal_threshold"]

print(f"Loaded model  | threshold={threshold:.4f} | features={len(feature_columns)}")

# ── 2. Prepare new data ────────────────────────────────────────────────────
def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a raw DataFrame with the same columns as the training dataset.
    Returns a feature-ready DataFrame aligned to training feature order.

    Drop leakage columns before calling this function, or they will be
    ignored automatically by the column selection below.
    """
    df = df_raw.copy()

    # Cast district
    if use_native_cat:
        df["district"] = df["district"].astype("category")
    else:
        df["district"] = le.transform(df["district"].astype(str))

    # Cast booleans to int
    for col in df.select_dtypes("bool").columns:
        df[col] = df[col].astype(int)

    # Select and order columns
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    return df[feature_columns]


# ── 3. Predict ─────────────────────────────────────────────────────────────
def predict_outbreak(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - outbreak_probability  : float [0, 1]
      - predicted_outbreak    : int   {0, 1}  (using optimal threshold)
    """
    X = prepare_features(df_raw)
    proba  = model.predict_proba(X)[:, 1]
    labels = (proba >= threshold).astype(int)

    return pd.DataFrame({
        "outbreak_probability": proba,
        "predicted_outbreak":   labels,
    }, index=df_raw.index)


# ── 4. Example usage ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Replace with actual new-week data (same schema as training CSV,
    # excluding target columns and leakage columns).
    try:
        sample = pd.read_csv("model_ready_district_week_trainable.csv")
        LEAKAGE_COLS = [
            "target_cases_plus1", "target_cases_plus2",
            "target_outbreak_plus2",
            "exclude_target_plus1", "exclude_target_plus2",
            "exclude_training_row",
            "target_outbreak_plus1",   # ← ground truth; not available at inference
        ]
        sample = sample.drop(columns=[c for c in LEAKAGE_COLS if c in sample.columns])

        results = predict_outbreak(sample.head(10))
        print("\nSample predictions:")
        print(results.to_string())
        outbreak_rate = results["predicted_outbreak"].mean()
        print(f"\nOutbreak rate in sample: {outbreak_rate:.1%}")

    except FileNotFoundError:
        print("Provide your new-week data as a DataFrame to predict_outbreak().")
