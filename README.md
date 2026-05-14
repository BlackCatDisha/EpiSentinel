# EpiSentinel: Technical Project Documentation

## 1. Project Overview
EpiSentinel is an advanced epidemic risk visualization and prediction dashboard designed for the state of Karnataka. It combines Machine Learning (Random Forest) with interactive geospatial visualizations (D3.js) to provide health officials with actionable insights into Dengue outbreaks.

---

## 2. File Architecture & Functionality

### **Frontend (Dashboard)**
| File | Responsibility |
| :--- | :--- |
| `index.html` | The structural foundation. Contains the grid layout, Sidebar, Map Container, Chatbot UI, and the District Detail Modal. |
| `style.css` | Implements a "Glassmorphism" design system. Handles responsiveness across mobile/desktop, sidebar animations, and risk-color tokens. |
| `app.js` | The logic engine. Handles GeoJSON fetching, D3.js map rendering, dynamic scaling, event listeners (hover/click), and the chatbot shell. |
| `data.js` | Contains the normalized prediction data and a name-mapping utility to sync GeoJSON district names with ML output names. |

### **Backend & Data Science**
| File | Responsibility |
| :--- | :--- |
| `predictions.json` | The bridge between ML and Frontend. Stores calculated risk scores, predicted cases, and thresholds for all districts. |
| `randomforest_model_eval.py` | Training and evaluation script. Calculates MAE (Mean Absolute Error) and F1-scores to validate model accuracy on historical data. |
| `randomforest_quantified_prediction.py` | The production script. It takes the latest available features and generates the final case counts and risk percentages. |

---

## 3. Dataset Analysis

### **Key Columns & Their Impact**
*   **Cases (Target)**: The number of reported Dengue cases. Used as the label for training.
*   **Temperature (Avg/Max/Min)**: Higher temperatures typically accelerate the mosquito life cycle.
*   **Humidity**: High humidity (>60%) is essential for mosquito survival and egg-laying.
*   **Rainfall**: Predicts stagnant water availability. However, excessive rainfall can "wash away" larvae (non-linear relationship).
*   **Lagged Cases (1-4 weeks)**: The strongest predictor. Outbreaks are often auto-regressive (current cases predict future cases).
*   **Population Density**: High-density urban areas (like Bengaluru Urban) show faster transmission rates.

---

## 4. Modeling Strategy: Ensemble vs. Single
For this project, an **Ensemble Model** is highly recommended over a single "one-size-fits-all" model.

*   **Why?** Random Forest (our current model) is great at handling non-linear relationships and missing data. However, adding **XGBoost** or **LightGBM** in a "Stacked Ensemble" would help capture sudden spikes that a single Random Forest might smoothen out.
*   **Benefit**: A combination reduces the variance of predictions, leading to more stable "Risk Scores" on the dashboard.

---

## 5. Roadmap: Live Data & 14-Day Lead Time

### **Predicting 14 Days in Advance**
Currently, we predict for the "Upcoming Week" (7-day lead). To extend this to 14 days:
1.  **Feature Shifting**: We shift our lagged features from `t-1` to `t-2`.
2.  **Reliability**: Accuracy will decrease slightly as weather forecasts become less certain beyond 7 days.

### **Linking Live Data**
To make EpiSentinel truly live, we must implement an automated pipeline:
*   **API Integration**: Use `requests` in Python to fetch weekly weather updates from APIs like OpenWeather.
*   **Syncing**: The Python script should run automatically on a server (e.g., AWS Lambda or GitHub Actions) to overwrite `predictions.json` every week.

---

## 6. Explainability Layer (XAI)
To build trust with medical professionals, we should add **SHAP (SHapley Additive exPlanations)**.
*   **The Goal**: When a user clicks on "Kolar" and sees "Critical Risk," the dashboard should explain: *"Risk is high due to a 30% spike in humidity and high case counts in neighboring districts last week."*
*   **Implementation**: We can calculate these "Importance Scores" in the Python script and include them in the `predictions.json` for the frontend to display.

---

## 7. Future Improvements
1.  **Spatial Correlation**: Incorporate "Neighboring District" risks (if District A is high, District B's risk increases).
2.  **Sentiment Analysis**: Scrape local news or social media for "fever" or "hospital" mentions as an early warning signal.
3.  **Mobile App**: Package the dashboard as a PWA (Progressive Web App) for field workers.
