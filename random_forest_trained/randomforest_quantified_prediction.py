import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, f1_score
import numpy as np


# ==============================
# 1. LOAD DATA
# ==============================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "model_ready_district_week_trainable.csv")
df = pd.read_csv(csv_path)

# ==============================
# 2. CLEAN DATA
# ==============================
df = df[df["exclude_training_row"] == 0]
df = df[df["exclude_target_plus1"] == 0]
df = df.dropna()

# ==============================
# 3. FEATURES
# ==============================
features = [
    "temperature_mean_week",
    "humidity_mean_week",
    "rainfall_total_week",
    "cases_lag1",
    "cases_lag2",
    "cases_lag3",
    "cases_roll2_mean",
    "cases_roll4_mean",
    "temperature_mean_week_lag1",
    "temperature_mean_week_lag2",
    "humidity_mean_week_lag1",
    "humidity_mean_week_lag2",
    "rainfall_total_week_lag1",
    "rainfall_total_week_lag2",
    "week_sin",
    "week_cos",
    "population_2011",
    "cases_per_100k"
]

target_class = "target_outbreak_plus1"
target_reg   = "target_cases_plus1"

# ==============================
# 4. TIME SPLIT
# ==============================
train = df[df["year"] <= 2021]
test  = df[df["year"] >= 2022]

X_train = train[features]
y_train_class = train[target_class]
y_train_reg   = train[target_reg]

X_test = test[features]
y_test_class = test[target_class]
y_test_reg   = test[target_reg]

# ==============================
# 5. TRAIN MODELS

# ==============================
print("Training Classification Model (Risk Score)...")
clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train_class)

print("Training Regression Model (Case Quantification)...")
reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
reg.fit(X_train, y_train_reg)

# ==============================
# 6. QUANTIFIED PREDICTIONS
# ==============================
print("\nGenerating Quantified Predictions...")
# Get probabilities for risk score
risk_scores = clf.predict_proba(X_test)[:, 1]
# Get numerical predictions for cases
case_preds = reg.predict(X_test)

# --- ACCURACY EVALUATION ---
# Calculate metrics on the test set
mae = mean_absolute_error(y_test_reg, case_preds)
# For F1, we use the best threshold from our other script or 0.4 as a standard
f1 = f1_score(y_test_class, (risk_scores > 0.4).astype(int))

# Create results dataframe
results = test.copy()
# Round cases to the nearest whole person (no decimals)
results["predicted_cases_next_week"] = case_preds.round().astype(int)
results["risk_score_percent"] = (risk_scores * 100).round(1)
results["outbreak_threshold"] = test["district_case_q75"]

# ==============================
# 7. DISTRICT-WISE SUMMARY (LATEST WEEK)
# ==============================
# Get the most recent week in the test set to show "current" status
latest_week = results.sort_values(["year", "iso_week"], ascending=False).drop_duplicates("district")

summary = latest_week[[
    "district", 
    "year", 
    "iso_week", 
    "outbreak_threshold", 
    "predicted_cases_next_week", 
    "risk_score_percent"
]].sort_values("risk_score_percent", ascending=False)

print("\n=== MODEL ACCURACY (VALIDATED ON 2022-2023 DATA) ===")
print(f"Case Prediction Error (MAE): {mae:.2f} cases")
print(f"Outbreak Detection (F1-Score): {f1:.2f}")

print("\n=== DISTRICT-WISE QUANTIFIED PREDICTIONS (LATEST WEEK) ===")
print("Note: These are raw weekly case counts (total people).")
print(summary.to_string(index=False))


# Save to JSON for the Dashboard (Best for Web Maps)
# We structure it as a dictionary with district names as keys for fast lookup
json_output = summary.set_index("district").to_dict(orient="index")
import json
with open(os.path.join(base_dir, "predictions.json"), "w") as f:
    json.dump(json_output, f, indent=4)

print(f"\nPredictions saved to: {os.path.join(base_dir, 'predictions.json')}")

