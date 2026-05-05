--- PROJECT: EpiSentinel ---

WHAT IT IS:An AI-powered dengue outbreak prediction system that gives district health officers a 7–14 day early warning before outbreaks peak, at block/ward level granularity across Indian districts.

THE CORE PROBLEM (rigorous version):District Health Officers and block-level health workers in high-burden Indian districts cannot make proactive resource allocation decisions — specifically deploying field teams, initiating vector control, and pre-positioning medical supplies — at block spatial resolution more than 3 days before a dengue outbreak peaks. This is because existing surveillance systems (IDSP, NVBDCP) operate with a 21-day retrospective reporting lag, provide no sub-district granularity, and integrate no environmental or climate signals. Every outbreak response is therefore reactive, missing the biological intervention window of 14–21 days that exists between heavy rainfall and clinical case onset.

WHY 14 DAYS IS THE RIGHT NUMBER:Dengue has a fixed biological clock:

Heavy rainfall → stagnant water → mosquito breeding: 7–10 days

Extrinsic incubation period in mosquito: 8–12 days

Human incubation after bite: 4–7 days

Total: ~14–21 days from rainfall to case surge

IDSP reporting lag on top: +21 days after cases appearSo the system needs to detect the signal (rainfall + environment) before the mosquito lifecycle completes.

HOW THE SYSTEM WORKS:

Every morning, APIs pull three types of data:

Climate: IMD (India Met Dept) and ERA5 for daily rainfall, temperature, humidity

Health: IDSP weekly dengue case reports (parsed from PDFs)

Satellite: Google Earth Engine for NDWI (water index) and NDVI (vegetation index) per ward

Feature engineering creates lag features (rainfall 7 days ago, cases 14 days ago), rolling averages, and a mosquito breeding index (NDWI × rainfall × temperature)

Two AI models run in parallel:

LSTM: reads a 30-day time window per ward, detects temporal trends and delayed effects

XGBoost: reads today's snapshot of features, matches conditions to historical outbreak patterns

Their scores are blended into one risk score (0–1) per ward per day

SHAP values from XGBoost explain why each ward is flagged in plain language

Output goes three ways: heatmap dashboard for DHOs, SMS alerts via Bhashini API in local languages for field workers, and surge forecasts for hospital admins

THE AI MODELS EXPLAINED SIMPLY:

LSTM (Long Short-Term Memory): A neural network that reads data sequentially day by day, maintains a memory cell, and is good at detecting patterns across time — e.g. "sustained rainfall 2 weeks ago predicts cases today." Like a doctor reading your diary for the past month.

XGBoost: 500 decision trees that each ask yes/no questions about today's data and vote. Each tree learns from the mistakes of the previous one (boosting). Good at feature interactions. Like 500 experienced doctors each with a checklist voting together.

SHAP: A mathematical method that looks inside XGBoost and calculates exactly how much each feature (rainfall, temperature, water index etc.) contributed to the final risk score. Converts black-box predictions into plain-language justifications.

TARGET USERS (personas):

DHO (District Health Officer): Strategic — sees full ward-level heatmap, SHAP explanations, resource deployment recommendations

NGO/Health Worker: Frontline — receives SMS in local language (Tamil, Hindi, Telugu) with block risk level and action

Hospital Admin: Clinical — sees 14-day bed/ICU demand forecast for surge planning
CURRENT SYSTEMS AND WHY THEY FAIL:

IDSP: 21-day lag, district-level only, no forecast, passive reporting

NVBDCP: Retrospective annual counts, no sub-district resolution, no prediction

Hospital EMRs: Only detect outbreak after surge begins — too late by definitionNone integrate climate signals. None produce forward-looking risk at block granularity.
