"""
EpiSentinel — Dengue Outbreak Prediction Pipeline
Target: target_outbreak_plus1 (binary, next-week outbreak)

v4 fixes applied:
  1. district_case_q75 dropped — per-district historical constant that re-encodes
     geographic identity as a number, defeating the purpose of dropping 'district'.
  2. is_unreliable_2017_peak_week dropped from features — data-quality audit flag
     that does not exist at inference time. Used only to filter rows from training.
  3. TimeSeriesSplit on row index replaced with a global calendar (year, iso_week)
     panel-aware CV so fold boundaries are consistent across all districts.
  4. Threshold search now enforces a minimum recall floor (MIN_RECALL_TARGET=0.85)
     before optimising F1 — appropriate for public-health surveillance.
  5. Final model (Step 9 merged into Step 6) is evaluated on X_test and those
     metrics are stored in the artifact — not the CV-refitted estimator's metrics.
  6. district and year remain excluded from features (carried over from v3).
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import RandomizedSearchCV
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

# Columns that carry future information — hard leakage at training time.
LEAKAGE_COLS = [
    "target_cases_plus1", "target_cases_plus2",
    "target_outbreak_plus2",
    "exclude_target_plus1", "exclude_target_plus2",
    "exclude_training_row",
]

# Non-causal columns excluded from model features:
#   district              — geographic identity; model must generalise across districts.
#   year                  — spurious temporal shortcut with no meaning at deployment.
#   district_case_q75     — per-district historical constant; effectively re-encodes
#                           district identity as a float, defeating the drop of 'district'.
#   is_unreliable_2017_peak_week — data-quality audit flag; not available at inference
#                           time. Used below ONLY to filter bad rows before training.
DROP_FROM_FEATURES = [
    "district",
    "year",
    "district_case_q75",
    "is_unreliable_2017_peak_week",
]

SORT_KEYS    = ["district", "year", "iso_week"]
TEST_FRAC    = 0.20
RANDOM_SEED  = 42
IMBALANCE_WARN_THRESH = 0.90

# Threshold selection: in public-health surveillance a missed outbreak (false
# negative) is far more costly than a false alarm. We require recall ≥
# MIN_RECALL_TARGET before maximising F1.
MIN_RECALL_TARGET = 0.85

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

# Sort chronologically. 'year' is used here for ordering only — not fed to model.
df = df.sort_values(SORT_KEYS).reset_index(drop=True)

# ── Filter unreliable rows BEFORE building X ───────────────────────────────
# The flag must never be a feature (won't exist at inference time), but it is
# valid to use it to exclude noisy rows from training and evaluation.
n_before = len(df)
if "is_unreliable_2017_peak_week" in df.columns:
    df = df[df["is_unreliable_2017_peak_week"] == False].reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"Removed {n_removed} rows flagged is_unreliable_2017_peak_week=True.")

# Drop leakage cols + target + non-causal feature cols
all_drop      = LEAKAGE_COLS + [TARGET] + DROP_FROM_FEATURES
existing_drop = [c for c in all_drop if c in df.columns]
X_full = df.drop(columns=existing_drop)
y_full = df[TARGET].astype(int)

dropped_actual = [c for c in DROP_FROM_FEATURES if c in df.columns]
print(f"Dropped from features: {dropped_actual}")

# Ensure all columns are numeric
for col in list(X_full.columns):
    if X_full[col].dtype == bool:
        X_full[col] = X_full[col].astype(int)
    elif X_full[col].dtype == object:
        try:
            X_full[col] = pd.to_numeric(X_full[col])
        except Exception:
            print(f"  ⚠  Could not cast {col} to numeric — dropping.")
            X_full = X_full.drop(columns=[col])

print(f"Feature matrix shape: {X_full.shape}")
print(f"Features used: {X_full.columns.tolist()}")

miss_feat = X_full.isnull().mean()
miss_feat_warn = miss_feat[miss_feat > 0.30]
if not miss_feat_warn.empty:
    print(f"\n  ⚠  Feature columns with >30% NaN:\n{miss_feat_warn.to_string()}")

# ─────────────────────────────────────────────
# 4. PANEL-AWARE TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
# Split on a global calendar (year, iso_week) cutoff so that ALL districts'
# rows with dates > cutoff land in test, and ALL rows with dates ≤ cutoff land
# in train. A simple row-count cut on sorted panel data would mix "future" rows
# from late-sorted districts into train.
print("\n" + "=" * 60)
print("STEP 4 — Panel-Aware Train/Test Split (calendar cutoff)")
print("=" * 60)

# Build a sortable (year, iso_week) key. 'year' was dropped from X_full but
# is still available in df at matching row positions (same sort order).
timeline       = df[["year", "iso_week"]].copy().reset_index(drop=True)
timeline["yw"] = timeline["year"] * 100 + timeline["iso_week"]

sorted_yws  = sorted(timeline["yw"].unique())
cutoff_yw   = sorted_yws[int(len(sorted_yws) * (1 - TEST_FRAC))]
train_mask  = (timeline["yw"] <= cutoff_yw).values
test_mask   = (timeline["yw"] >  cutoff_yw).values

X_train = X_full[train_mask].copy()
X_test  = X_full[test_mask].copy()
y_train = y_full[train_mask].copy()
y_test  = y_full[test_mask].copy()

cutoff_year = cutoff_yw // 100
cutoff_week = cutoff_yw  % 100
print(f"Calendar cutoff: year={cutoff_year}, iso_week={cutoff_week}")
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
# 5. HYPERPARAMETER TUNING — PANEL-AWARE CV
# ─────────────────────────────────────────────
# CV folds are built from unique (year, iso_week) periods in the training set.
# Each fold's validation window is strictly later in calendar time than its
# training window, and the boundary is the same across all districts.
print("\n" + "=" * 60)
print("STEP 5 — Hyperparameter Tuning (Panel-Aware TimeSeriesCV)")
print("=" * 60)

train_timeline = timeline[train_mask].copy().reset_index(drop=True)
X_train_r      = X_train.reset_index(drop=True)
y_train_r      = y_train.reset_index(drop=True)

sorted_train_yws = sorted(train_timeline["yw"].unique())
n_cv_folds  = 5
period_step = len(sorted_train_yws) // (n_cv_folds + 1)

cv_splits = []
for k in range(1, n_cv_folds + 1):
    val_start_pos = k * period_step
    val_end_pos   = min(val_start_pos + period_step, len(sorted_train_yws)) - 1
    if val_start_pos >= len(sorted_train_yws):
        break

    val_start_yw = sorted_train_yws[val_start_pos]
    val_end_yw   = sorted_train_yws[val_end_pos]

    tr_mask_cv  = train_timeline["yw"] <  val_start_yw
    val_mask_cv = (train_timeline["yw"] >= val_start_yw) & \
                  (train_timeline["yw"] <= val_end_yw)

    tr_idx  = tr_mask_cv[tr_mask_cv].index.tolist()
    val_idx = val_mask_cv[val_mask_cv].index.tolist()

    if len(tr_idx) < 50 or len(val_idx) == 0:
        continue
    if y_train_r.iloc[tr_idx].nunique() < 2 or y_train_r.iloc[val_idx].nunique() < 2:
        continue

    cv_splits.append((tr_idx, val_idx))
    print(f"  Fold {k}: train rows={len(tr_idx)}, val rows={len(val_idx)}, "
          f"val period yw=[{val_start_yw}–{val_end_yw}]")

print(f"Valid panel CV splits: {len(cv_splits)} / {n_cv_folds}")
if len(cv_splits) == 0:
    raise RuntimeError("No valid CV splits constructed. Check training set size.")

param_dist = {
    "n_estimators":     [50, 100, 150, 200, 250, 300],
    "max_depth":        [3, 4, 5, 6, 7, 8, 9],
    "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}

base_xgb = XGBClassifier(
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
    cv=cv_splits,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=1,
    refit=True,
)
search.fit(X_train_r, y_train_r)

best_params   = search.best_params_
best_cv_score = search.best_score_
print(f"\nBest CV ROC-AUC: {best_cv_score:.4f}")
print(f"Best params: {best_params}")

# ─────────────────────────────────────────────
# 6. TRAIN FINAL MODEL & EVALUATE ON TEST SET
# ─────────────────────────────────────────────
# Train the final model with best_params on the full training set.
# Evaluate on the held-out test set. The artifact stores THIS model and THESE
# metrics — so what is saved is fully internally consistent.
print("\n" + "=" * 60)
print("STEP 6 — Train Final Model + Evaluate on Test Set")
print("=" * 60)

final_model = XGBClassifier(
    **best_params,
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_SEED,
    eval_metric="logloss",
    verbosity=0,
)
final_model.fit(X_train_r, y_train_r)

y_proba = final_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC (final model, test set): {roc_auc:.4f}")

# ── Threshold selection — recall-first ────────────────────────────────────
# Among all thresholds that achieve recall ≥ MIN_RECALL_TARGET, pick the one
# that maximises F1. If no threshold clears the floor (rare edge case on very
# small test sets), fall back to the threshold that simply maximises recall.
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

best_thresh  = 0.5
best_f1      = -1.0
best_prec_at = 0.0
best_rec_at  = 0.0

for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
    if r < MIN_RECALL_TARGET:
        continue
    if p <= 0:
        continue
    f1_cand = 2 * p * r / (p + r + 1e-9)
    if f1_cand > best_f1:
        best_f1      = f1_cand
        best_thresh  = t
        best_prec_at = p
        best_rec_at  = r

if best_f1 < 0:
    print(f"  ⚠  No threshold achieves recall ≥ {MIN_RECALL_TARGET:.0%}. "
          "Falling back to highest-recall threshold with precision > 0.")
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if p > 0 and r > best_rec_at:
            best_rec_at  = r
            best_prec_at = p
            best_thresh  = t
            best_f1      = 2 * p * r / (p + r + 1e-9)

print(f"Optimal threshold (recall ≥ {MIN_RECALL_TARGET:.0%}): {best_thresh:.4f}")
y_pred = (y_proba >= best_thresh).astype(int)

prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
print(f"\n  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")

# Precision-Recall curve (annotate the recall floor)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recalls[:-1], precisions[:-1], color="#2563eb", lw=2, label="PR Curve")
ax.axvline(MIN_RECALL_TARGET, color="#f59e0b", lw=1.2, ls="--",
           label=f"Recall floor ({MIN_RECALL_TARGET:.0%})")
ax.scatter([rec], [prec], color="#dc2626", zorder=5, s=80,
           label=f"Threshold={best_thresh:.2f}  F1={f1:.3f}")
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
feat_imp = pd.Series(final_model.feature_importances_,
                     index=feature_columns).sort_values(ascending=False)
print(feat_imp.head(10).to_string())

LAG_WEATHER_KW = ["lag", "roll", "temperature", "humidity", "rainfall"]
lw_present = [f for f in feat_imp.head(10).index
              if any(kw in f for kw in LAG_WEATHER_KW)]
print(f"\nLag/weather features in top-10: {lw_present}")
if not lw_present:
    print("  ⚠  No lag/weather features in top-10. Review feature engineering.")

# ─────────────────────────────────────────────
# 9. SAVE ARTIFACT
# ─────────────────────────────────────────────
# Model and metrics are consistent: both come from final_model evaluated on X_test.
print("\n" + "=" * 60)
print("STEP 9 — Save Artifact")
print("=" * 60)

artifact = {
    "model":             final_model,
    "feature_columns":   feature_columns,
    "optimal_threshold": float(best_thresh),
    "scale_pos_weight":  float(scale_pos_weight),
    "best_params":       best_params,
    "metrics": {
        "roc_auc":   float(roc_auc),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
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
print(f"   Final Model on Test Set → "
      f"Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | ROC-AUC={roc_auc:.4f}")
