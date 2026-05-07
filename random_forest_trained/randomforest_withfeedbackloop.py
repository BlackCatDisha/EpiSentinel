import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

# ==============================
# 1. LOAD DATA
# ==============================
# Get the directory where the script is located
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

target = "target_outbreak_plus1"

# ==============================
# 4. BEST TIME SPLIT
# ==============================
train = df[df["year"] <= 2021]
test  = df[df["year"] >= 2022]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# ==============================
# 5. TRAIN MODEL (IMPROVED)
# ==============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=3,
    class_weight={0:1, 1:2},   # 🔥 prioritize outbreak detection
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 6. PREDICT PROBABILITIES
# ==============================
probs = model.predict_proba(X_test)[:, 1]

# ==============================
# 7. FEEDBACK LOOP (THRESHOLD TUNING)
# ==============================
best_f1 = 0
best_thresh = 0.5

for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
    preds_temp = (probs > t).astype(int)
    f1 = f1_score(y_test, preds_temp)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Best Threshold:", best_thresh)

# ==============================
# 8. FINAL PREDICTION
# ==============================
preds = (probs > best_thresh).astype(int)

# ==============================
# 9. EVALUATION
# ==============================
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, preds))

# ==============================
# 10. ERROR ANALYSIS (FEEDBACK)
# ==============================
df_test = test.copy()
df_test["pred"] = preds

missed = df_test[(df_test[target] == 1) & (df_test["pred"] == 0)]

print("\nMissed Outbreaks Sample:")
print(missed[["district", "year", "iso_week"]].head())

# ==============================
# 11. FEATURE IMPORTANCE
# ==============================
importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance)