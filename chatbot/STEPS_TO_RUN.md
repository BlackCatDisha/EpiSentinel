#Environment file
Create .env in the project root:
bashGOOGLE_API_KEY=your_gemini_api_key_here
CONTEXT_MD_PATH=context.md
GEMINI_MODEL=gemini-1.5-flash
Get your Gemini API key from aistudio.google.com.


#Install dependencies from requirements.txt


#Run the server
bashuvicorn main:app --reload --port 8000
You should see:
INFO: context.md loaded from 'context.md' (XXXX chars).
INFO: Uvicorn running on http://127.0.0.1:8000
Open the auto-generated docs at http://localhost:8000/docs — both endpoints will be listed there and are fully interactive.

#Test with curl/postman
District Health Officer:
bashcurl -X POST http://localhost:8000/chat/district \
  -H "Content-Type: application/json" \
  -d '{
    "role": "district_health_officer",
    "district_name": "Raichur",
    "risk_score": 0.84,
    "predicted_cases": 47,
    "shap_drivers": [
      {
        "feature": "cases_roll4_mean",
        "display_name": "4-week rolling average of reported cases",
        "shap_value": 0.72,
        "feature_value": 18.5
      },
      {
        "feature": "rainfall_total_week_lag1",
        "display_name": "total rainfall last week",
        "shap_value": 0.41,
        "feature_value": 97.6
      }
    ],
    "user_message": "What immediate actions should I take?"
  }'
Hospital Manager:
bashcurl -X POST http://localhost:8000/chat/district \
  -H "Content-Type: application/json" \
  -d '{
    "role": "hospital_manager",
    "district_name": "Raichur",
    "risk_score": 0.84,
    "predicted_cases": 47,
    "shap_drivers": [
      {
        "feature": "cases_roll4_mean",
        "display_name": "4-week rolling average of reported cases",
        "shap_value": 0.72,
        "feature_value": 18.5
      }
    ],
    "user_message": "How many beds and supplies should I prepare?"
  }'
State Official:
bashcurl -X POST http://localhost:8000/chat/state \
  -H "Content-Type: application/json" \
  -d '{
    "role": "state_official",
    "aggregates": {
      "total_predicted_cases": 312,
      "average_risk_score": 0.61,
      "active_alerts": 7
    },
    "districts": [
      {
        "district_name": "Raichur",
        "risk_score": 0.84,
        "predicted_cases": 47,
        "shap_drivers": [
          {
            "feature": "cases_roll4_mean",
            "display_name": "4-week rolling average of reported cases",
            "shap_value": 0.72,
            "feature_value": 18.5
          }
        ]
      },
      {
        "district_name": "Vijayapura",
        "risk_score": 0.21,
        "predicted_cases": 3,
        "shap_drivers": [
          {
            "feature": "iso_week",
            "display_name": "week of the year",
            "shap_value": -0.31,
            "feature_value": 12.0
          }
        ]
      }
    ],
    "user_message": "Which districts need state-level intervention this week?"
  }'
Note that Vijayapura (risk 0.21) will be excluded from the LLM prompt detail and only counted in the aggregate, exactly as designed.

Test the guardrails
These should trigger polite refusals, not answers:
bash# Out-of-domain question
curl -X POST http://localhost:8000/chat/district \
  -H "Content-Type: application/json" \
  -d '{
    "role": "district_health_officer",
    "district_name": "Raichur",
    "risk_score": 0.5,
    "predicted_cases": 10,
    "shap_drivers": [{"feature": "iso_week", "display_name": "week of year", "shap_value": 0.1, "feature_value": 32}],
    "user_message": "Can you write me a Python script to sort a list?"
  }'

# Question outside the SOPs
curl -X POST http://localhost:8000/chat/district \
  -H "Content-Type: application/json" \
  -d '{
    "role": "district_health_officer",
    "district_name": "Raichur",
    "risk_score": 0.5,
    "predicted_cases": 10,
    "shap_drivers": [{"feature": "iso_week", "display_name": "week of year", "shap_value": 0.1, "feature_value": 32}],
    "user_message": "What is the recommended antiviral dosage for dengue treatment?"
  }'