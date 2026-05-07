"""
EpiSentinel — Human-Readable SHAP Explanations
Generates per-prediction natural language explanations + a summary report.
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import textwrap

# ── Load artifacts ─────────────────────────────────────────────────────────
artifact        = joblib.load("episentinel_pipeline.joblib")
model           = artifact["model"]
feature_columns = artifact["feature_columns"]
district_map    = artifact["district_map"]
threshold       = artifact["optimal_threshold"]

# Reverse map for display: int → district name
district_map_inv = {v: k for k, v in district_map.items()}

# ── Load & prepare data ────────────────────────────────────────────────────
df = pd.read_csv("model_ready_district_week_trainable.csv")

LEAKAGE_COLS = [
    "target_cases_plus1", "target_cases_plus2", "target_outbreak_plus2",
    "exclude_target_plus1", "exclude_target_plus2",
    "exclude_training_row", "target_outbreak_plus1",
]
SORT_KEYS = ["district", "year", "iso_week"]

# Keep original district name for display before encoding
df = df.sort_values(SORT_KEYS).reset_index(drop=True)
df["_district_name"] = df["district"]   # save for display

X_full = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])
X_full["district"] = X_full["district"].map(district_map).astype(np.int64)
for col in X_full.select_dtypes("bool").columns:
    X_full[col] = X_full[col].astype(int)

split_idx = int(len(X_full) * 0.8)
X_train   = X_full.iloc[:split_idx][feature_columns].copy().astype(np.float64)

# Use test set for explanations (unseen data)
X_test_raw = X_full.iloc[split_idx:][feature_columns].copy().astype(np.float64)
meta_test  = df.iloc[split_idx:][["_district_name", "year", "iso_week"]].reset_index(drop=True)

# ── Compute SHAP values ────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
X_explain   = X_test_raw.sample(200, random_state=42).reset_index(drop=True)
meta_explain = meta_test.loc[X_explain.index].reset_index(drop=True)

shap_values = explainer(X_explain, check_additivity=False)
probas      = model.predict_proba(X_explain)[:, 1]
predictions = (probas >= threshold).astype(int)

# ── Feature display name map ───────────────────────────────────────────────
DISPLAY_NAMES = {
    "district_case_q75":          "district's historical 75th-percentile case load",
    "cases_roll4_mean":           "4-week rolling average of reported cases",
    "cases_roll2_mean":           "2-week rolling average of reported cases",
    "cases_lag1":                 "cases reported last week",
    "cases_lag2":                 "cases reported 2 weeks ago",
    "cases_lag3":                 "cases reported 3 weeks ago",
    "iso_week":                   "week of the year",
    "year":                       "year",
    "district":                   "district identity",
    "population_2011":            "district population",
    "dengue_cases_reported":      "dengue cases reported this week",
    "cases_per_100k":             "cases per 100,000 population",
    "temperature_mean_week":      "average temperature this week",
    "temperature_max_week":       "maximum temperature this week",
    "temperature_mean_week_lag1": "average temperature last week",
    "temperature_mean_week_lag2": "average temperature 2 weeks ago",
    "humidity_mean_week":         "average humidity this week",
    "humidity_max_week":          "maximum humidity this week",
    "humidity_mean_week_lag1":    "average humidity last week",
    "humidity_mean_week_lag2":    "average humidity 2 weeks ago",
    "rainfall_total_week":        "total rainfall this week",
    "rainfall_total_week_lag1":   "total rainfall last week",
    "rainfall_total_week_lag2":   "total rainfall 2 weeks ago",
    "week_sin":                   "seasonal cycle (sine component)",
    "week_cos":                   "seasonal cycle (cosine component)",
    "dengue_deaths_weekly":       "dengue deaths this week",
    "is_unreliable_2017_peak_week": "data reliability flag (2017 peak)",
}

def display_name(feat):
    return DISPLAY_NAMES.get(feat, feat.replace("_", " "))

# ── Per-prediction explanation ─────────────────────────────────────────────
def explain_prediction(idx, shap_vals, feature_vals, proba, pred,
                        district_name, year, week, top_n=5):
    """Return a human-readable string explaining one prediction."""

    sv   = shap_vals.values[idx]          # SHAP values for this row
    fv   = feature_vals.iloc[idx]         # actual feature values

    # Sort features by absolute SHAP impact
    order       = np.argsort(np.abs(sv))[::-1]
    top_feats   = [(feature_columns[i], sv[i], fv.iloc[i]) for i in order[:top_n]]

    verdict     = "⚠️  OUTBREAK PREDICTED" if pred == 1 else "✅  NO OUTBREAK PREDICTED"
    conf_word   = ("high" if abs(proba - 0.5) > 0.25
                   else "moderate" if abs(proba - 0.5) > 0.1
                   else "low")

    lines = [
        f"{'─'*65}",
        f"District : {district_name}   |   Year: {year}   |   Week: {week}",
        f"Verdict  : {verdict}",
        f"Outbreak probability : {proba:.1%}  ({conf_word} confidence)",
        f"",
        f"Why did the model make this prediction?",
    ]

    push_up   = [(f, s, v) for f, s, v in top_feats if s > 0]
    push_down = [(f, s, v) for f, s, v in top_feats if s < 0]

    if push_up:
        lines.append("  Factors that INCREASED outbreak risk:")
        for feat, shap_val, feat_val in push_up:
            # Format feature value sensibly
            if feat == "district":
                val_str = district_name
            elif isinstance(feat_val, float) and feat_val != int(feat_val):
                val_str = f"{feat_val:.2f}"
            else:
                val_str = str(int(feat_val)) if not np.isnan(feat_val) else "N/A"
            lines.append(
                f"    + {display_name(feat)} was {val_str}"
                f"  →  pushed risk up by {shap_val:+.3f}"
            )

    if push_down:
        lines.append("  Factors that DECREASED outbreak risk:")
        for feat, shap_val, feat_val in push_down:
            if feat == "district":
                val_str = district_name
            elif isinstance(feat_val, float) and feat_val != int(feat_val):
                val_str = f"{feat_val:.2f}"
            else:
                val_str = str(int(feat_val)) if not np.isnan(feat_val) else "N/A"
            lines.append(
                f"    - {display_name(feat)} was {val_str}"
                f"  →  pushed risk down by {shap_val:+.3f}"
            )

    return "\n".join(lines)


# ── Global feature summary ─────────────────────────────────────────────────
def global_summary(shap_vals, top_n=10):
    mean_abs = np.abs(shap_vals.values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:top_n]

    lines = [
        "=" * 65,
        "GLOBAL MODEL BEHAVIOUR — What drives outbreak predictions overall?",
        "=" * 65,
    ]
    for rank, i in enumerate(order, 1):
        feat     = feature_columns[i]
        impact   = mean_abs[i]
        # Direction: do high values tend to push toward or away from outbreak?
        sv_col   = shap_vals.values[:, i]
        fv_col   = shap_vals.data[:, i]
        corr     = np.corrcoef(fv_col, sv_col)[0, 1] if np.std(fv_col) > 0 else 0
        direction = "↑ higher value → more risk" if corr > 0.05 \
                    else "↓ higher value → less risk" if corr < -0.05 \
                    else "↕ mixed / non-linear effect"
        lines.append(
            f"  {rank:>2}. {display_name(feat):<45}"
            f"  avg impact={impact:.3f}   {direction}"
        )
    return "\n".join(lines)


# ── Run explanations ───────────────────────────────────────────────────────
print(global_summary(shap_values))
print()

# Print explanations for: top 3 highest-risk + top 3 lowest-risk predictions
sorted_by_proba = np.argsort(probas)
highlight_idx   = list(sorted_by_proba[-3:][::-1]) + list(sorted_by_proba[:3])
labels          = ["[HIGH RISK #1]", "[HIGH RISK #2]", "[HIGH RISK #3]",
                   "[LOW RISK  #1]", "[LOW RISK  #2]", "[LOW RISK  #3]"]

print("=" * 65)
print("SAMPLE PREDICTIONS — Detailed Explanations")
print("=" * 65)

all_explanations = []
for label, idx in zip(labels, highlight_idx):
    print(f"\n{label}")
    exp = explain_prediction(
        idx, shap_values, X_explain, probas[idx], predictions[idx],
        meta_explain.loc[idx, "_district_name"],
        int(meta_explain.loc[idx, "year"]),
        int(meta_explain.loc[idx, "iso_week"]),
    )
    print(exp)
    all_explanations.append(f"{label}\n{exp}")

# ── Save full explanation report ───────────────────────────────────────────
with open("shap_explanations.txt", "w") as f:
    f.write(global_summary(shap_values))
    f.write("\n\n")
    f.write("=" * 65 + "\n")
    f.write("SAMPLE PREDICTIONS — Detailed Explanations\n")
    f.write("=" * 65 + "\n\n")
    for label, exp in zip(labels, all_explanations):
        f.write(exp + "\n\n")

print("\n\nFull explanation report saved → shap_explanations.txt")
print("✅  Done.")
