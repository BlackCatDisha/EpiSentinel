"""
EpiSentinel — Human-Readable SHAP Explanations  (v4, no district / year / district_case_q75)
"""

import joblib
import pandas as pd
import numpy as np
import shap

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

# Preserve metadata columns for human-readable output (not fed to the model).
df["_district_name"] = df["district"]
df["_year"]          = df["year"]
df["_iso_week"]      = df["iso_week"]

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

X_test_raw = X_full[test_mask][feature_columns].reset_index(drop=True).astype(np.float64)
meta_test  = df[test_mask][["_district_name", "_year", "_iso_week"]].reset_index(drop=True)

# ── SHAP on a sample of the test set ──────────────────────────────────────
n_explain    = min(200, len(X_test_raw))
sample_idx   = X_test_raw.sample(n_explain, random_state=42).index
X_explain    = X_test_raw.loc[sample_idx].reset_index(drop=True)
meta_explain = meta_test.loc[sample_idx].reset_index(drop=True)

explainer    = shap.TreeExplainer(model)
shap_values  = explainer(X_explain, check_additivity=False)
probas       = model.predict_proba(X_explain)[:, 1]
predictions  = (probas >= threshold).astype(int)

# ── Feature display names ──────────────────────────────────────────────────
# Note: district_case_q75, year, district, and is_unreliable_2017_peak_week
# are intentionally absent — they are not model features in v4.
DISPLAY_NAMES = {
    "cases_roll4_mean":           "4-week rolling average of reported cases",
    "cases_roll2_mean":           "2-week rolling average of reported cases",
    "cases_lag1":                 "cases reported last week",
    "cases_lag2":                 "cases reported 2 weeks ago",
    "cases_lag3":                 "cases reported 3 weeks ago",
    "dengue_cases_reported":      "dengue cases reported this week",
    "cases_per_100k":             "cases per 100,000 population",
    "iso_week":                   "week of the year",
    "population_2011":            "district population",
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
}

def display_name(feat):
    return DISPLAY_NAMES.get(feat, feat.replace("_", " "))

def fmt_val(feat, val):
    if np.isnan(val):
        return "N/A"
    if feat in ("iso_week", "population_2011"):
        return str(int(val))
    return f"{val:.2f}"

# ── Per-prediction explanation ─────────────────────────────────────────────
def explain_prediction(idx, top_n=5):
    sv   = shap_values.values[idx]
    fv   = X_explain.iloc[idx]
    prob = probas[idx]
    pred = predictions[idx]

    district = meta_explain.loc[idx, "_district_name"]
    year     = int(meta_explain.loc[idx, "_year"])
    week     = int(meta_explain.loc[idx, "_iso_week"])

    order     = np.argsort(np.abs(sv))[::-1]
    top_feats = [(feature_columns[i], sv[i], fv.iloc[i]) for i in order[:top_n]]

    verdict   = "⚠️  OUTBREAK PREDICTED" if pred == 1 else "✅  NO OUTBREAK PREDICTED"
    conf_word = ("high"     if abs(prob - 0.5) > 0.25 else
                 "moderate" if abs(prob - 0.5) > 0.10 else "low")

    lines = [
        "─" * 65,
        f"District : {district}   |   Year: {year}   |   Week: {week}",
        f"Verdict  : {verdict}",
        f"Outbreak probability : {prob:.1%}  ({conf_word} confidence)",
        "",
        "Why did the model make this prediction?",
    ]

    push_up   = [(f, s, v) for f, s, v in top_feats if s > 0]
    push_down = [(f, s, v) for f, s, v in top_feats if s < 0]

    if push_up:
        lines.append("  Factors that INCREASED outbreak risk:")
        for feat, shap_val, feat_val in push_up:
            lines.append(
                f"    + {display_name(feat)} was {fmt_val(feat, feat_val)}"
                f"  →  pushed risk up by {shap_val:+.3f}"
            )
    if push_down:
        lines.append("  Factors that DECREASED outbreak risk:")
        for feat, shap_val, feat_val in push_down:
            lines.append(
                f"    - {display_name(feat)} was {fmt_val(feat, feat_val)}"
                f"  →  pushed risk down by {shap_val:+.3f}"
            )

    return "\n".join(lines)

# ── Global summary ─────────────────────────────────────────────────────────
def global_summary(top_n=10):
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:top_n]

    lines = [
        "=" * 65,
        "GLOBAL MODEL BEHAVIOUR",
        "What drives outbreak predictions overall?",
        "=" * 65,
    ]
    for rank, i in enumerate(order, 1):
        feat      = feature_columns[i]
        impact    = mean_abs[i]
        sv_col    = shap_values.values[:, i]
        fv_col    = shap_values.data[:, i]
        corr      = np.corrcoef(fv_col, sv_col)[0, 1] if np.std(fv_col) > 0 else 0
        direction = ("↑ higher value → more outbreak risk"  if corr >  0.05 else
                     "↓ higher value → less outbreak risk"  if corr < -0.05 else
                     "↕ mixed / non-linear effect")
        lines.append(
            f"  {rank:>2}. {display_name(feat):<48}"
            f"avg impact={impact:.3f}   {direction}"
        )
    return "\n".join(lines)

# ── Run & print ────────────────────────────────────────────────────────────
summary = global_summary()
print(summary)
print()

sorted_by_proba = np.argsort(probas)
highlight_idx   = list(sorted_by_proba[-3:][::-1]) + list(sorted_by_proba[:3])
labels = [
    "[HIGH RISK #1]", "[HIGH RISK #2]", "[HIGH RISK #3]",
    "[LOW RISK  #1]", "[LOW RISK  #2]", "[LOW RISK  #3]",
]

print("=" * 65)
print("SAMPLE PREDICTIONS — Detailed Explanations")
print("=" * 65)

all_explanations = []
for label, idx in zip(labels, highlight_idx):
    print(f"\n{label}")
    exp = explain_prediction(idx)
    print(exp)
    all_explanations.append(f"{label}\n{exp}")

# ── Save report ────────────────────────────────────────────────────────────
with open("shap_explanations.txt", "w", encoding="utf-8") as f:
    f.write(summary + "\n\n")
    f.write("=" * 65 + "\n")
    f.write("SAMPLE PREDICTIONS — Detailed Explanations\n")
    f.write("=" * 65 + "\n\n")
    for exp in all_explanations:
        f.write(exp + "\n\n")

print("\n\nFull report saved → shap_explanations.txt")
print("✅  Done.")
