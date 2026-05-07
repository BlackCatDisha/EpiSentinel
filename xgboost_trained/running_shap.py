import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ── Load artifacts ─────────────────────────────────────────────────────────
artifact        = joblib.load("episentinel_pipeline.joblib")
model           = artifact["model"]
feature_columns = artifact["feature_columns"]
district_map    = artifact["district_map"]

# ── Load & sort dataset ────────────────────────────────────────────────────
df = pd.read_csv("model_ready_district_week_trainable.csv")

LEAKAGE_COLS = [
    "target_cases_plus1", "target_cases_plus2", "target_outbreak_plus2",
    "exclude_target_plus1", "exclude_target_plus2",
    "exclude_training_row", "target_outbreak_plus1",
]
SORT_KEYS = ["district", "year", "iso_week"]

df = df.sort_values(SORT_KEYS).reset_index(drop=True)
X_full = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

# Apply the same district encoding used at training time
X_full["district"] = X_full["district"].map(district_map).astype(np.int64)

# Booleans → int
for col in X_full.select_dtypes("bool").columns:
    X_full[col] = X_full[col].astype(int)

# ── Match train split & align features ────────────────────────────────────
split_idx = int(len(X_full) * 0.8)
X_train   = X_full.iloc[:split_idx][feature_columns].copy()

# ── Sample ────────────────────────────────────────────────────────────────
X_sample   = X_train.sample(500, random_state=42).astype(np.float64)
background = X_sample.sample(100, random_state=0).astype(np.float64)

# ── SHAP ───────────────────────────────────────────────────────────────────
# With integer-encoded district (no enable_categorical), TreeExplainer
# works with both tree_path_dependent and interventional modes.
# We use tree_path_dependent here (default, no background needed, faster)
# since the categorical incompatibility is now gone.
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_sample, check_additivity=False)

# ── Plots ──────────────────────────────────────────────────────────────────
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("EpiSentinel — SHAP Beeswarm (Top 15 Features)")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → shap_beeswarm.png")

plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("EpiSentinel — SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → shap_bar.png")

print("\n✅ SHAP analysis complete.")
