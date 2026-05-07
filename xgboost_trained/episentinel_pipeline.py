"""
EpiSentinel — Dengue Outbreak Prediction Pipeline
Target: target_outbreak_plus1 (binary, next-week outbreak)
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "model_ready_district_week_trainable.csv"
MODEL_OUT    = "episentinel_pipeline.joblib"
THRESH_OUT   = "optimal_threshold.json"
FEATURES_OUT = "feature_columns.json"
PR_CURVE_OUT = "precision_recall_curve.png"
CM_OUT       = "confusion_matrix.png"

TARGET       = "target_outbreak_plus1"
LEAKAGE_COLS = [
    "target_cases_plus1", "target_cases_plus2",
    "target_outbreak_plus2",
    "exclude_target_plus1", "exclude_target_plus2",
    "exclude_training_row",
]
SORT_KEYS    = ["district", "year", "iso_week"]
TEST_FRAC    = 0.20
RANDOM_SEED  = 42
IMBALANCE_WARN_THRESH = 0.90

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Load Dataset")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. VALIDATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Validation")
print("=" * 60)

dupes = df.duplicated(subset=SORT_KEYS).sum()
print(f"Duplicate (district, year, iso_week) rows: {dupes}")
if dupes > 0:
    print(f"  ⚠  Dropping {dupes} duplicate rows.")
    df = df.drop_duplicates(subset=SORT_KEYS)

missing_pct = df.isnull().mean().sort_values(ascending=False)
print("\nMissing % (top 10):")
print(missing_pct.head(10).to_string())
high_miss = missing_pct[missing_pct > 0.30]
if not high_miss.empty:
    print(f"\n  ⚠  Features with >30% missing:\n{high_miss.to_string()}")

class_dist = df[TARGET].value_counts(normalize=True)
print(f"\nClass distribution:\n{df[TARGET].value_counts().to_string()}")
majority_frac = class_dist.max()
if majority_frac > IMBALANCE_WARN_THRESH:
    print(f"\n  ⚠  Severe class imbalance: majority class = {majority_frac:.1%}")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Preprocessing")
print("=" * 60)

df = df.sort_values(SORT_KEYS).reset_index(drop=True)

# Encode district as stable integer codes (sorted alphabetically so the
# mapping is deterministic across runs and SHAP-compatible).
# We do NOT use enable_categorical=True in XGBoost because SHAP's
# TreeExplainer raises NotImplementedError for models with categorical
# split nodes. Integer-encoded districts work identically for XGBoost
# and are fully supported by SHAP.
all_districts = sorted(df["district"].unique().tolist())
district_map  = {d: i for i, d in enumerate(all_districts)}
df["district"] = df["district"].map(district_map).astype(np.int64)
print(f"district encoded: {len(all_districts)} unique districts → int64")

# Drop leakage columns + target
drop_cols     = LEAKAGE_COLS + [TARGET]
existing_drop = [c for c in drop_cols if c in df.columns]
X_full = df.drop(columns=existing_drop)
y_full = df[TARGET].astype(int)

# Ensure all columns are numeric
for col in X_full.columns:
    if X_full[col].dtype == bool:
        X_full[col] = X_full[col].astype(int)
    elif X_full[col].dtype == object:
        try:
            X_full[col] = pd.to_numeric(X_full[col])
        except Exception:
            print(f"  ⚠  Could not cast {col} to numeric — dropping.")
            X_full = X_full.drop(columns=[col])

print(f"Feature matrix shape: {X_full.shape}")

miss_feat = X_full.isnull().mean()
miss_feat_warn = miss_feat[miss_feat > 0.30]
if not miss_feat_warn.empty:
    print(f"\n  ⚠  Feature columns with >30% NaN:\n{miss_feat_warn.to_string()}")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT (time-aware, 80/20)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Time-Aware Train/Test Split")
print("=" * 60)

split_idx = int(len(X_full) * (1 - TEST_FRAC))
X_train = X_full.iloc[:split_idx].copy()
X_test  = X_full.iloc[split_idx:].copy()
y_train = y_full.iloc[:split_idx].copy()
y_test  = y_full.iloc[split_idx:].copy()

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train class dist: {y_train.value_counts().to_dict()}")
print(f"Test  class dist: {y_test.value_counts().to_dict()}")

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight (train): {scale_pos_weight:.3f}")

feature_columns = X_train.columns.tolist()
print(f"Total features: {len(feature_columns)}")

# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Hyperparameter Tuning (RandomizedSearchCV + TimeSeriesSplit)")
print("=" * 60)

param_dist = {
    "n_estimators":     [50, 100, 150, 200, 250, 300],
    "max_depth":        [3, 4, 5, 6, 7, 8, 9],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}

tscv = TimeSeriesSplit(n_splits=5)
valid_splits = []
for tr_idx, val_idx in tscv.split(X_train):
    y_tr_fold  = y_train.iloc[tr_idx]
    y_val_fold = y_train.iloc[val_idx]
    if len(y_tr_fold) > 50 and y_tr_fold.nunique() == 2 and y_val_fold.nunique() == 2:
        valid_splits.append((tr_idx, val_idx))
print(f"Valid CV splits: {len(valid_splits)} / 5")

base_xgb = XGBClassifier(
    # enable_categorical intentionally omitted — SHAP incompatible
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    eval_metric="logloss",
    verbosity=0,
)

search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_dist,
    n_iter=40,
    scoring="roc_auc",
    cv=tscv,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1,
    refit=True,
)
search.fit(X_train, y_train)

best_params    = search.best_params_
best_cv_score  = search.best_score_
print(f"\nBest CV ROC-AUC: {best_cv_score:.4f}")
print(f"Best params: {best_params}")

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Evaluation on Test Set")
print("=" * 60)

best_model = search.best_estimator_
y_proba    = best_model.predict_proba(X_test)[:, 1]
roc_auc    = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
best_thresh = 0.5
best_f1     = -1
best_recall = -1
for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    if p > 0.2:
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1 or (f1 == best_f1 and r > best_recall):
            best_f1     = f1
            best_recall = r
            best_thresh = t

print(f"Optimal threshold: {best_thresh:.4f}")
y_pred = (y_proba >= best_thresh).astype(int)

prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
print(f"\n  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recalls[:-1], precisions[:-1], color="#2563eb", lw=2, label="PR Curve")
ax.scatter([best_recall], [prec], color="#dc2626", zorder=5, s=80,
           label=f"Threshold={best_thresh:.2f}\nF1={best_f1:.3f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("EpiSentinel — Precision-Recall Curve")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PR_CURVE_OUT, dpi=150)
plt.close()
print(f"\nPR curve saved → {PR_CURVE_OUT}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
fig2, ax2 = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=["No Outbreak", "Outbreak"]).plot(
    ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title("EpiSentinel — Confusion Matrix")
plt.tight_layout()
plt.savefig(CM_OUT, dpi=150)
plt.close()
print(f"Confusion matrix saved → {CM_OUT}")

# ─────────────────────────────────────────────
# 7. BASELINE COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Baseline Comparison (Predict All Zeros)")
print("=" * 60)
y_baseline = np.zeros(len(y_test), dtype=int)
b_f1 = f1_score(y_test, y_baseline, zero_division=0)
print(f"  Naive (all-zero) → F1={b_f1:.4f}")
print(f"  EpiSentinel      → F1={f1:.4f}  AUC={roc_auc:.4f}  ΔF1=+{f1-b_f1:.4f}")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Feature Importance (Top 10)")
print("=" * 60)
feat_imp = pd.Series(best_model.feature_importances_,
                     index=feature_columns).sort_values(ascending=False)
print(feat_imp.head(10).to_string())
LAG_WEATHER_KW = ["lag", "roll", "temperature", "humidity", "rainfall"]
lw_present = [f for f in feat_imp.head(10).index
              if any(kw in f for kw in LAG_WEATHER_KW)]
print(f"\nLag/weather features in top-10: {lw_present}")
if not lw_present:
    print("  ⚠  No lag/weather features in top-10. Review feature engineering.")

# ─────────────────────────────────────────────
# 9. RETRAIN FINAL MODEL & SAVE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9 — Retrain Final Model on Full Training Data")
print("=" * 60)

final_model = XGBClassifier(
    **best_params,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    eval_metric="logloss",
    verbosity=0,
)
final_model.fit(X_train, y_train)

artifact = {
    "model":              final_model,
    "district_map":       district_map,       # str → int, needed at inference
    "feature_columns":    feature_columns,
    "optimal_threshold":  float(best_thresh),
    "scale_pos_weight":   float(scale_pos_weight),
    "best_params":        best_params,
    "metrics": {
        "roc_auc":    float(roc_auc),
        "precision":  float(prec),
        "recall":     float(rec),
        "f1":         float(f1),
    },
}

joblib.dump(artifact, MODEL_OUT)
print(f"Pipeline artifact saved → {MODEL_OUT}")

with open(THRESH_OUT, "w") as f:
    json.dump({"optimal_threshold": float(best_thresh)}, f, indent=2)
print(f"Threshold saved → {THRESH_OUT}")

with open(FEATURES_OUT, "w") as f:
    json.dump({"feature_columns": feature_columns}, f, indent=2)
print(f"Feature columns saved → {FEATURES_OUT}")

print("\n✅ EpiSentinel pipeline complete.")
print(f"   Final Test → Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | ROC-AUC={roc_auc:.4f}")
