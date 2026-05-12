const districtData = {
    "Uttara Kannada": { "predicted_cases": 7, "risk_score": 80.1, "threshold": 15.0, "status": "Critical", "top_driver": "Rainfall Lag-1 & Humidity", "detailed_explanation": "Ensemble (XGBoost 49% + CatBoost 30% + RF 9% + Poisson 11% + LightGBM 1%): Lagged Rainfall Total (lag-1) increased risk by 18.4% | Humidity (lag-2) elevated seasonal transmission | Population Density contributed moderate upward pressure. ROC-AUC: 0.821, Recall: 85.6%" },
    "Bengaluru Rural": { "predicted_cases": 4, "risk_score": 79.2, "threshold": 15.0, "status": "Critical", "top_driver": "4-Week Case Rolling Average", "detailed_explanation": "Ensemble Model: 4-week rolling case average (cases_roll4_mean) is the dominant predictor increasing risk by 14.2% | Temperature lag-1 contributed +8.1% | NDWI (water body index) elevated vector habitat risk. Model Recall: 85.6%" },
    "Kolar": { "predicted_cases": 5, "risk_score": 77.7, "threshold": 15.0, "status": "Critical", "top_driver": "4-Week Case Rolling Average", "detailed_explanation": "Ensemble Model: High 4-week rolling case average drives risk upward by 12.1% | Seasonal sine component (week_sin) indicates peak transmission window | Temperature lag-1 adds +7.3% pressure. Model F1: 0.720" },
    "Chitradurga": { "predicted_cases": 12, "risk_score": 74.1, "threshold": 15.0, "status": "Critical", "top_driver": "4-Week Case Rolling Average", "detailed_explanation": "Ensemble Model: 4-week outbreak momentum (14.25 avg cases) increased risk by 7.7% | Seasonal transmission window (week_sin) elevated probability by 5.0% | 2-week case velocity confirms active spread. Recommended: Intensive Surveillance" },
    "Mandya": { "predicted_cases": 11, "risk_score": 69.3, "threshold": 3.0, "status": "High", "top_driver": "Total Population Scale", "detailed_explanation": "Outbreak Momentum (4-week avg) (12.25) increased risk by 8.0% | Recent Case Velocity (2-week avg) (16.00) increased risk by 5.2% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 4.4%" },
    "Raichur": { "predicted_cases": 7, "risk_score": 68.5, "threshold": 0.0, "status": "High", "top_driver": "Total Population Scale", "detailed_explanation": "Total Population Scale (1,849,004) increased risk by 18.2% | Local Population Density (218) increased risk by 3.3% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 1.8%" },
    "Shivamogga": { "predicted_cases": 12, "risk_score": 65.7, "threshold": 12.0, "status": "High", "top_driver": "Total Population Scale", "detailed_explanation": "Outbreak Momentum (4-week avg) (15.00) increased risk by 6.7% | Recent Case Velocity (2-week avg) (17.50) increased risk by 5.8% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 3.5%" },
    "Tumakuru": { "predicted_cases": 5, "risk_score": 64.4, "threshold": 3.25, "status": "High", "top_driver": "Total Population Scale", "detailed_explanation": "Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 4.9% | Outbreak Momentum (4-week avg) (8.75) increased risk by 4.6% | Total Population Scale (2,387,006) increased risk by 4.0%" },
    "Chamarajanagara": { "predicted_cases": 8, "risk_score": 64.0, "threshold": 4.0, "status": "High", "top_driver": "Total Population Scale", "detailed_explanation": "Recent Case Velocity (2-week avg) (13.00) increased risk by 7.1% | Outbreak Momentum (4-week avg) (9.00) increased risk by 5.8% | Total Population Scale (925,954) decreased risk by 4.8%" },
    "Vijayapura": { "predicted_cases": 10, "risk_score": 57.0, "threshold": 6.0, "status": "Medium", "top_driver": "Total Population Scale", "detailed_explanation": "Recent Case Velocity (2-week avg) (6.50) increased risk by 3.8% | Total Population Scale (2,240,568) increased risk by 2.7% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 2.5%" },
    "Belagavi": { "predicted_cases": 12, "risk_score": 56.7, "threshold": 2.0, "status": "Medium", "top_driver": "Total Population Scale", "detailed_explanation": "Total Population Scale (4,660,682) decreased risk by 5.9% | Recent Case Velocity (2-week avg) (13.00) increased risk by 5.1% | Outbreak Momentum (4-week avg) (11.25) increased risk by 4.5%" },
    "Bidar": { "predicted_cases": 5, "risk_score": 54.9, "threshold": 1.0, "status": "Medium", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (291) decreased risk by 3.8% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 3.7% | Outbreak Momentum (4-week avg) (4.75) increased risk by 3.4%" },
    "Bagalkote": { "predicted_cases": 6, "risk_score": 51.7, "threshold": 3.0, "status": "Medium", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (270) decreased risk by 4.4% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 2.9% | Historical Case Load (3 Weeks Ago) (8.00) increased risk by 2.8%" },
    "Kodagu": { "predicted_cases": 5, "risk_score": 51.3, "threshold": 1.0, "status": "Medium", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (121) increased risk by 11.6% | Total Population Scale (499,403) decreased risk by 8.1% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 2.5%" },
    "Hassan": { "predicted_cases": 5, "risk_score": 50.5, "threshold": 3.25, "status": "Medium", "top_driver": "Total Population Scale", "detailed_explanation": "Outbreak Momentum (4-week avg) (5.75) increased risk by 5.9% | Local Population Density (231) decreased risk by 5.2% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 4.4%" },
    "Davangere": { "predicted_cases": 6, "risk_score": 48.9, "threshold": 15.0, "status": "Medium", "top_driver": "Seasonal Transmission Cycle", "detailed_explanation": "Ensemble Model: Seasonal sine component (week_sin) increased risk by 3.1% | Population scale (2.1M) adds moderate upward pressure | Low recent case velocity (3.0 avg) keeps risk contained. Routine monitoring advised." },
    "Mysuru": { "predicted_cases": 12, "risk_score": 48.8, "threshold": 7.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (473) decreased risk by 13.6% | Outbreak Momentum (4-week avg) (13.75) increased risk by 6.1% | Total Population Scale (2,994,709) decreased risk by 5.9%" },
    "Kalaburagi": { "predicted_cases": 12, "risk_score": 46.9, "threshold": 6.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Total Population Scale (3,693,906) decreased risk by 5.5% | Outbreak Momentum (4-week avg) (28.00) increased risk by 5.2% | Recent Case Velocity (2-week avg) (17.50) increased risk by 4.3%" },
    "Chikkamagaluru": { "predicted_cases": 5, "risk_score": 44.6, "threshold": 3.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Total Population Scale (1,003,996) decreased risk by 7.8% | Local Population Density (139) decreased risk by 3.6% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 3.5%" },
    "Dharwad": { "predicted_cases": 6, "risk_score": 43.8, "threshold": 0.25, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (418) decreased risk by 10.4% | Outbreak Momentum (4-week avg) (5.75) increased risk by 2.8% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 2.5%" },
    "Ballari": { "predicted_cases": 8, "risk_score": 42.4, "threshold": 5.25, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (300) decreased risk by 5.5% | Total Population Scale (2,540,928) increased risk by 4.7% | Previous Week Temperature (25.15) decreased risk by 2.9%" },
    "Koppal": { "predicted_cases": 6, "risk_score": 41.8, "threshold": 5.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (238) decreased risk by 5.1% | Total Population Scale (1,331,777) decreased risk by 4.0% | Historical Case Load (3 Weeks Ago) (9.00) increased risk by 3.0%" },
    "Haveri": { "predicted_cases": 7, "risk_score": 38.5, "threshold": 6.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (285) decreased risk by 6.4% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 2.7% | Total Population Scale (1,373,132) decreased risk by 2.4%" },
    "Gadag": { "predicted_cases": 6, "risk_score": 35.2, "threshold": 5.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Total Population Scale (1,041,286) decreased risk by 7.8% | Local Population Density (224) decreased risk by 4.6% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 2.0%" },
    "Bengaluru Urban": { "predicted_cases": 4, "risk_score": 35.0, "threshold": 1.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (5,223) decreased risk by 7.0% | Total Population Scale (11,424,837) decreased risk by 6.0% | Seasonal Transmission Cycle (Sine) (-0.24) increased risk by 2.7%" },
    "Dakshina Kannada": { "predicted_cases": 12, "risk_score": 34.9, "threshold": 10.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Local Population Density (435) decreased risk by 9.6% | Temperature Trend (2 Weeks Ago) (26.59) decreased risk by 5.0% | Current Average Temperature (26.14) decreased risk by 3.6%" },
    "Udupi": { "predicted_cases": 9, "risk_score": 19.1, "threshold": 8.0, "status": "Low", "top_driver": "Total Population Scale", "detailed_explanation": "Total Population Scale (1,063,433) decreased risk by 8.1% | Local Population Density (272) decreased risk by 7.9% | Previous Week Temperature (26.85) decreased risk by 4.9%" },
    // Missing districts from model results, adding placeholders
    "Chikkaballapura": { "predicted_cases": 0, "risk_score": 0, "threshold": 0, "status": "Low", "top_driver": "Data Missing", "detailed_explanation": "No predictive data available for this district in the 2023 test set." },
    "Ramanagara": { "predicted_cases": 0, "risk_score": 0, "threshold": 0, "status": "Low", "top_driver": "Data Missing", "detailed_explanation": "No predictive data available for this district in the 2023 test set." },
    "Yadgir": { "predicted_cases": 0, "risk_score": 0, "threshold": 0, "status": "Low", "top_driver": "Data Missing", "detailed_explanation": "No predictive data available for this district in the 2023 test set." }
};

// Utility to match GeoJSON names with our data names
const nameAliasMap = {
    "Bangalore": "Bengaluru Urban",
    "Bangalore Urban": "Bengaluru Urban",
    "Bangalore Rural": "Bengaluru Rural",
    "Chikkamaggaluru": "Chikkamagaluru",
    "Mysore": "Mysuru",
    "Belgaum": "Belagavi",
    "Gulbarga": "Kalaburagi",
    "Bellary": "Ballari",
    "Bijapur": "Vijayapura",
    "Chikmagalur": "Chikkamagaluru",
    "Shimoga": "Shivamogga",
    "Bagalkot": "Bagalkote",
    "Chamrajnagar": "Chamarajanagara",
    "Chamarajanagar": "Chamarajanagara",
    // FIX: GeoJSON spells it "Davanagere" (3 a's), our data uses "Davangere"
    "Davanagere": "Davangere",
    "Davangere": "Davangere",
    "Gadag": "Gadag",
    "Haveri": "Haveri",
    "Uttara Kannada": "Uttara Kannada",
    "Dakshina Kannada": "Dakshina Kannada",
    "Udupi": "Udupi",
    "Kodagu": "Kodagu",
    "Kolar": "Kolar",
    "Mandya": "Mandya",
    "Tumkur": "Tumakuru",
    "Andaman and Nicobar Islands": "Andaman & Nicobar",
    "Dadra and Nagar Haveli and Daman and Diu": "D&N Haveli & Daman & Diu",
    "Jammu and Kashmir": "Jammu & Kashmir",
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand"
};

const stateData = {
    "Karnataka": {
        "predicted_cases": Object.values(districtData).reduce((acc, curr) => acc + curr.predicted_cases, 0),
        "risk_score": Math.round(Object.values(districtData).reduce((acc, curr) => acc + curr.risk_score, 0) / Object.values(districtData).length * 10) / 10,
        "status": "High"
    },
    "Maharashtra": { "predicted_cases": 245, "risk_score": 68.4, "status": "High" },
    "Kerala": { "predicted_cases": 182, "risk_score": 75.2, "status": "Critical" },
    "Tamil Nadu": { "predicted_cases": 156, "risk_score": 54.8, "status": "Medium" },
    "Andhra Pradesh": { "predicted_cases": 98, "risk_score": 42.1, "status": "Low" },
    "Telangana": { "predicted_cases": 112, "risk_score": 48.5, "status": "Low" },
    "Gujarat": { "predicted_cases": 134, "risk_score": 51.2, "status": "Medium" },
    "Rajasthan": { "predicted_cases": 87, "risk_score": 38.6, "status": "Low" },
    "Madhya Pradesh": { "predicted_cases": 105, "risk_score": 45.3, "status": "Low" },
    "Uttar Pradesh": { "predicted_cases": 210, "risk_score": 62.7, "status": "High" },
    "Bihar": { "predicted_cases": 145, "risk_score": 58.4, "status": "Medium" },
    "West Bengal": { "predicted_cases": 167, "risk_score": 65.9, "status": "High" },
    "Odisha": { "predicted_cases": 78, "risk_score": 35.2, "status": "Low" },
    "Punjab": { "predicted_cases": 64, "risk_score": 32.8, "status": "Low" },
    "Haryana": { "predicted_cases": 52, "risk_score": 30.1, "status": "Low" },
    "Delhi": { "predicted_cases": 128, "risk_score": 72.4, "status": "High" },
    "Assam": { "predicted_cases": 89, "risk_score": 47.6, "status": "Low" },
    "Uttarakhand": { "predicted_cases": 45, "risk_score": 34.2, "status": "Low" },
    "Himachal Pradesh": { "predicted_cases": 32, "risk_score": 28.5, "status": "Low" },
    "Jammu & Kashmir": { "predicted_cases": 41, "risk_score": 31.7, "status": "Low" },
    "Jharkhand": { "predicted_cases": 72, "risk_score": 41.8, "status": "Low" },
    "Chhattisgarh": { "predicted_cases": 68, "risk_score": 39.4, "status": "Low" },
    "Goa": { "predicted_cases": 15, "risk_score": 25.6, "status": "Low" }
};

function normalizeName(name) {
    if (!name) return name;
    if (nameAliasMap[name]) return nameAliasMap[name];
    
    const lowerName = name.toLowerCase();
    // Check aliases case-insensitively
    for (const key in nameAliasMap) {
        if (key.toLowerCase() === lowerName) return nameAliasMap[key];
    }
    
    // Check stateData keys case-insensitively
    for (const key in stateData) {
        if (key.toLowerCase() === lowerName) return key;
    }

    return name;
}

// Generate dummy district data for other states
function generateDummyDistricts(features) {
    const dummyData = {};
    features.forEach(f => {
        const name = f.properties.district || f.properties.NAME_2 || f.properties.name;
        const risk = 20 + Math.random() * 60;
        let status = "Low";
        if (risk > 70) status = "Critical";
        else if (risk > 50) status = "Medium";

        dummyData[normalizeName(name)] = {
            "predicted_cases": Math.floor(Math.random() * 20),
            "risk_score": Math.round(risk * 10) / 10,
            "status": status,
            "top_driver": ["High Temperature", "Heavy Rainfall", "Stagnant Water", "Population Density"][Math.floor(Math.random() * 4)],
            "detailed_explanation": "Environmental factors and historical trends contribute to this risk level."
        };
    });
    return dummyData;
}
