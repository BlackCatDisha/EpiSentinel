# EpiSentinel: Technical Project Documentation

## 1. Project Overview
EpiSentinel is an AI-assisted dengue early warning and response platform for public health teams.

It helps officials:
- See outbreak risk clearly on an interactive map
- Prioritize districts based on predicted risk and case load
- Understand key risk drivers behind each prediction
- Get fast, role-aware advisory guidance for field and policy actions

Detailed district-level intelligence is currently available for Karnataka, while other states are shown in preview mode.

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
| `chatbot/main.py` | FastAPI app entrypoint, CORS setup, router inclusion, and static dashboard hosting at `/dashboard`. |
| `chatbot/router.py` | Chat API endpoints: `/chat/general`, `/chat/district`, `/chat/state`. |
| `chatbot/predict_router.py` | CSV upload inference endpoint at `POST /predict`; tries weighted ensemble first, then XGBoost fallback. |
| `chatbot/episentinel_chatbot.py` | Prompt construction, role-specific schemas, SOP grounding (`context.md`), and Gemini invocation via LangChain. |
| `dashboard/data.js` | Current district/state data used directly by the frontend map and local chatbot fallback. |
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
For this project, the backend inference path is ensemble-first with fallback logic.

*   `POST /predict` first attempts to load `final_model_plus1_no_district_pop_standardised/weighted_ensemble_plus1_near_optimal_no_district_pop.joblib`.
*   If ensemble artifacts/dependencies are unavailable, it falls back to `xgboost_trained/episentinel_pipeline.joblib`.
*   This gives resilient startup behavior while preserving a stronger default model path.

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
Explainability is currently implemented in two forms:
*   **Dashboard-level explanations** come from precomputed text in `dashboard/data.js` (`top_driver`, `detailed_explanation`).
*   **`/predict` endpoint responses** include a fast proxy explanation by selecting the highest-magnitude feature for each row (not full per-request SHAP for latency reasons).

The repository also includes SHAP-related scripts/artifacts under model directories for offline analysis.

---

## 7. Chatbot Feature

EpiSentinel includes an in-dashboard AI assistant designed for real public-health workflows.

### What the chatbot helps with
- Interpreting district risk and predicted case trends
- Explaining likely outbreak drivers in plain language
- Recommending practical next actions for surveillance and response
- Answering dengue preparedness questions for rapid decision-making

### Role-aware guidance
The advisory flow is tailored for:
- District Health Officers
- Hospital Managers
- State Health Officials

Each role gets recommendations aligned to operational priorities, from district-level action plans to state-level resource coordination.

### Grounded, policy-aware responses
Chat responses are guided by context provided by us so recommendations stay aligned with public-health protocols and escalation practices.

---

## 8. Future Improvements
1.  **Spatial Correlation**: Incorporate "Neighboring District" risks (if District A is high, District B's risk increases).
2.  **Sentiment Analysis**: Scrape local news or social media for "fever" or "hospital" mentions as an early warning signal.
3.  **Mobile App**: Package the dashboard as a PWA (Progressive Web App) for field workers.

---

## 9. First-Time Setup and Run (New System)

Follow these steps if you are running EpiSentinel for the first time on a fresh machine.

### Prerequisites
- Python 3.10+ installed
- `pip` available
- A Gemini API key from https://aistudio.google.com/

### Step 1: Clone and enter the project
```bash
git clone <your-repo-url>
cd EpiSentinel
```

### Step 2: Create and activate a virtual environment
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r chatbot/requirements.txt
```

### Step 4: Create environment file for the backend
Create `chatbot/.env` with:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
CONTEXT_MD_PATH=context.md
GEMINI_MODEL=gemini-2.5-flash
```

Notes:
- `CONTEXT_MD_PATH=context.md` works because the backend is run from inside the `chatbot` folder.
- Keep the API key private and never commit `.env` to Git.

### Step 5: Start the backend API + dashboard host
```bash
cd chatbot
uvicorn main:app --reload --port 8000
```

If startup is successful, open:
- Dashboard: http://127.0.0.1:8000/dashboard/
- API docs: http://127.0.0.1:8000/docs



### Common issues
- `GOOGLE_API_KEY not set`: verify `chatbot/.env` exists and backend was started from the `chatbot` directory.
- Model load fallback warning: the app tries ensemble first, then falls back to XGBoost if ensemble dependencies/artifacts are unavailable.
