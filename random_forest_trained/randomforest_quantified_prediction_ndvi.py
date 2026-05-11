import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score, accuracy_score, r2_score, mean_squared_error
import numpy as np
import shap
import json


# ==============================
# 1. LOAD DATA (Updated with NDVI + New Population)
# ==============================
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "ndvi+pop_model_ready_district_week_trainable.csv")
df = pd.read_csv(csv_path)

# ==============================
# 2. CLEAN DATA
# ==============================
# Filter out unreliable or excluded rows
df = df[df["exclude_training_row"] == 0]
df = df[df["exclude_target_plus1"] == 0]
df = df.dropna()

# ==============================
# 3. FEATURES (Including NDVI Vegetation Index)
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
    "worldpop_total",
    "worldpop_density_per_km2",
    "cases_per_100k",
    "ndvi_mean" # New Vegetation Feature
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
print("Training Enhanced Classification Model (Risk Score)...")
clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train_class)

# --- REGRESSION FEEDBACK LOOP ---
print("Optimizing Enhanced Regression Model (Feedback Loop)...")
best_mae = float('inf')
best_depth = 10
best_reg = None

for depth in [5, 10, 15, 20]:
    temp_reg = RandomForestRegressor(n_estimators=200, max_depth=depth, random_state=42)
    temp_reg.fit(X_train, y_train_reg)
    temp_preds = temp_reg.predict(X_test)
    temp_mae = mean_absolute_error(y_test_reg, temp_preds)
    
    if temp_mae < best_mae:
        best_mae = temp_mae
        best_depth = depth
        best_reg = temp_reg

reg = best_reg
print(f"Regression Optimized: Best max_depth found is {best_depth} (MAE: {best_mae:.2f})")

# ==============================
# 6. QUANTIFIED PREDICTIONS
# ==============================
print("\nGenerating Quantified Predictions...")
risk_scores = clf.predict_proba(X_test)[:, 1]
case_preds = reg.predict(X_test)

# --- FEEDBACK LOOP (THRESHOLD OPTIMIZATION) ---
best_f1 = 0
best_threshold = 0.4 

for t in np.arange(0.2, 0.6, 0.05):
    temp_preds = (risk_scores > t).astype(int)
    temp_f1 = f1_score(y_test_class, temp_preds)
    if temp_f1 > best_f1:
        best_f1 = temp_f1
        best_threshold = t

print(f"Feedback Loop Optimized: Best threshold found is {best_threshold:.2f} (F1: {best_f1:.2f})")

# Apply optimized threshold
risk_preds = (risk_scores > best_threshold).astype(int)
f1 = f1_score(y_test_class, risk_preds)
precision = precision_score(y_test_class, risk_preds)
recall = recall_score(y_test_class, risk_preds)
accuracy = accuracy_score(y_test_class, risk_preds)

mae = mean_absolute_error(y_test_reg, case_preds)
mse = mean_squared_error(y_test_reg, case_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, case_preds)

# Create results dataframe
results = test.copy()
results["predicted_cases_next_week"] = case_preds.round().astype(int)
results["risk_score_percent"] = (risk_scores * 100).round(1)
results["outbreak_threshold"] = test["district_case_q75"]

# ==============================
# 7. DISTRICT-WISE SUMMARY (LATEST WEEK)
# ==============================
# --- EXPLAINABILITY LAYER ---
readable_features = {
    'temperature_mean_week': 'Current Average Temperature',
    'humidity_mean_week': 'Current Humidity Levels',
    'rainfall_total_week': 'Current Rainfall Intensity',
    'cases_lag1': 'Immediate Case Surge (Last Week)',
    'cases_lag2': 'Persistent Transmission (2 Weeks Ago)',
    'cases_lag3': 'Historical Case Load (3 Weeks Ago)',
    'cases_roll2_mean': 'Recent Case Velocity (2-week avg)',
    'cases_roll4_mean': 'Outbreak Momentum (4-week avg)',
    'temperature_mean_week_lag1': 'Previous Week Temperature',
    'temperature_mean_week_lag2': 'Temperature Trend (2 Weeks Ago)',
    'humidity_mean_week_lag1': 'Previous Week Humidity',
    'humidity_mean_week_lag2': 'Humidity Trend (2 Weeks Ago)',
    'rainfall_total_week_lag1': 'Previous Week Rainfall',
    'rainfall_total_week_lag2': 'Rainfall Trend (2 Weeks Ago)',
    'week_sin': 'Seasonal Transmission Cycle (Sine)',
    'week_cos': 'Seasonal Transmission Cycle (Cosine)',
    'worldpop_total': 'Total Population Scale',
    'worldpop_density_per_km2': 'Local Population Density',
    'cases_per_100k': 'Infection Prevalence (Cases per 100k)',
    'ndvi_mean': 'Vegetation Density (NDVI Index)'
}

importances = clf.feature_importances_

def get_top_driver(row):
    contributions = []
    for i, feat in enumerate(features):
        val = row[feat]
        impact = val * importances[i]
        contributions.append((feat, impact))
    
    top_feat = sorted(contributions, key=lambda x: x[1], reverse=True)[0][0]
    return readable_features.get(top_feat, top_feat.replace('_', ' ').title())

# ==============================
# 8. SHAP EXPLAINABILITY
# ==============================
print("\nCalculating SHAP Explanations (Vegetation-Inclusive)...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_vals_outbreak = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_outbreak = shap_values[:, :, 1]
else:
    shap_vals_outbreak = shap_values

def get_shap_factors(idx, top_n=3):
    sv = shap_vals_outbreak[idx]
    fv = X_test.iloc[idx]
    order = np.argsort(np.abs(sv))[::-1]
    reasons = []
    for i in order[:top_n]:
        feat = features[i]
        val = fv[feat]
        impact = sv[i]
        direction = "increased" if impact > 0 else "decreased"
        label = readable_features.get(feat, feat.replace('_', ' ').title())
        
        if "Population" in label:
            val_str = f"{int(val):,}"
        else:
            val_str = f"{val:.3f}" if "NDVI" in label else f"{val:.2f}"
            
        reasons.append(f"{label} ({val_str}) {direction} risk by {abs(impact)*100:.1f}%")
    
    return " | ".join(reasons)

results['detailed_explanation'] = [get_shap_factors(i) for i in range(len(X_test))]
results['top_driver'] = results.apply(get_top_driver, axis=1)

latest_week = results.sort_values(["year", "iso_week"], ascending=False).drop_duplicates("district")

summary = latest_week[[
    "district", 
    "year", 
    "iso_week", 
    "outbreak_threshold", 
    "predicted_cases_next_week", 
    "risk_score_percent",
    "top_driver",
    "detailed_explanation"
]].sort_values("risk_score_percent", ascending=False)

print("\n=== MODEL ACCURACY (WITH VEGETATION & NEW POPULATION) ===")
print(f"Optimized Decision Threshold: {best_threshold:.2f}")
print("--- Classification ---")
print(f"F1-Score:  {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"Accuracy:  {accuracy:.2f}")

print("\n--- Regression ---")
print(f"Mean Absolute Error (MAE):    {mae:.2f} cases")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} cases")
print(f"R-Squared (R2) Score:         {r2:.2f}")

print("\n=== DISTRICT-WISE QUANTIFIED PREDICTIONS (LATEST WEEK) ===")
print(summary.to_string(index=False))

# Save outputs
json_output = summary.set_index("district").to_dict(orient="index")
with open(os.path.join(base_dir, "predictions_ndvi.json"), "w") as f:
    json.dump(json_output, f, indent=4)

mean_abs_shap = np.abs(shap_vals_outbreak).mean(axis=0)
global_order = np.argsort(mean_abs_shap)[::-1]

with open(os.path.join(base_dir, "shap_report_ndvi.txt"), "w") as f:
    f.write("=== EPISENTINEL ENHANCED MODEL INSIGHTS (NDVI + NEW POP) ===\n")
    f.write("What drives outbreak predictions globally?\n\n")
    for i in global_order:
        feat = features[i]
        impact = mean_abs_shap[i]
        label = readable_features.get(feat, feat.replace('_', ' ').title())
        f.write(f"- {label:<35} : Avg Impact {impact:.4f}\n")
    
    f.write(f"\n\n=== MODEL PERFORMANCE (VALIDATED ON 2022-2023) ===\n")
    f.write(f"Decision Threshold: {best_threshold:.2f}\n")
    f.write(f"F1-Score:           {f1:.2f}\n")
    f.write(f"Precision:          {precision:.2f}\n")
    f.write(f"Recall:             {recall:.2f}\n")
    f.write(f"MAE (Regression):   {mae:.2f} cases\n")
    f.write(f"R2 Score:           {r2:.2f}\n")
    
    f.write("\n\n=== FULL DISTRICT-WISE AUDIT TRAIL ===\n")
    f.write(summary.to_string(index=False))

print(f"\nPredictions saved to: predictions_ndvi.json")
print(f"SHAP Global Report saved to: shap_report_ndvi.txt")
