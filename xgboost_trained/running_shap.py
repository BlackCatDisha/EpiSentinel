"""
EpiSentinel — SHAP Beeswarm + Bar Charts  (v4, no district / year / district_case_q75)
Generates visual SHAP summaries using the same test-set split as the pipeline.
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ── Load artifacts ─────────────────────────────────────────────────────────
artifact        = joblib.load("episentinel_pipeline.joblib")
model           = artifact["model"]
feature_columns = artifact["feature_columns"]
threshold       = artifact["optimal_threshold"]

# ── Load & prepare data ────────────────────────────────────────────────────
df = pd.read_csv("model_ready_district_week_trainable.csv")

# Columns excluded from the model — must match episentinel_pipeline.py exactly.
LEAKAGE_COLS = [
    "target_cases_plus1", "target_cases_plus2", "target_outbreak_plus2",
    "exclude_target_plus1", "exclude_target_plus2",
    "exclude_training_row", "target_outbreak_plus1",
    # Non-causal feature columns (same DROP_FROM_FEATURES as training):
    "district",                    # geographic identity
    "year",                        # spurious temporal shortcut
    "district_case_q75",           # per-district constant re-encoding district identity
    "is_unreliable_2017_peak_week",# audit flag — not available at inference time
]
SORT_KEYS = ["district", "year", "iso_week"]

df = df.sort_values(SORT_KEYS).reset_index(drop=True)

# ── Apply the same row filter as training ─────────────────────────────────
if "is_unreliable_2017_peak_week" in df.columns:
    df = df[df["is_unreliable_2017_peak_week"] == False].reset_index(drop=True)

X_full = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
for col in X_full.select_dtypes("bool").columns:
    X_full[col] = X_full[col].astype(int)

# ── Reproduce the panel-aware calendar split from the pipeline ─────────────
timeline       = df[["year", "iso_week"]].copy().reset_index(drop=True)
timeline["yw"] = timeline["year"] * 100 + timeline["iso_week"]
sorted_yws     = sorted(timeline["yw"].unique())
TEST_FRAC      = 0.20
cutoff_yw      = sorted_yws[int(len(sorted_yws) * (1 - TEST_FRAC))]
test_mask      = (timeline["yw"] > cutoff_yw).values

# ── Sample from X_test (consistent with shap_explain.py) ──────────────────
X_test = X_full[test_mask][feature_columns].reset_index(drop=True).astype(np.float64)
X_sample = X_test.sample(min(500, len(X_test)), random_state=42).reset_index(drop=True)

print(f"SHAP sample: {len(X_sample)} rows from test set "
      f"(cutoff yw={cutoff_yw}, test rows={test_mask.sum()})")

# ── SHAP ───────────────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_sample, check_additivity=False)

# ── Beeswarm ───────────────────────────────────────────────────────────────
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("EpiSentinel — SHAP Beeswarm (Top 15 Features)")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → shap_beeswarm.png")

# ── Bar ────────────────────────────────────────────────────────────────────
plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("EpiSentinel — SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → shap_bar.png")

print("\n✅ SHAP analysis complete.")
