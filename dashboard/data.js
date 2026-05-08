const districtData = {
    "Kolar": { "predicted_cases": 7, "risk_score": 80.2, "threshold": 0.0, "status": "Critical" },
    "Mysuru": { "predicted_cases": 11, "risk_score": 72.6, "threshold": 7.0, "status": "High" },
    "Chitradurga": { "predicted_cases": 11, "risk_score": 69.5, "threshold": 8.0, "status": "High" },
    "Chamarajanagara": { "predicted_cases": 7, "risk_score": 67.9, "threshold": 4.0, "status": "High" },
    "Kodagu": { "predicted_cases": 3, "risk_score": 67.5, "threshold": 1.0, "status": "High" },
    "Belagavi": { "predicted_cases": 10, "risk_score": 66.6, "threshold": 2.0, "status": "High" },
    "Bengaluru Rural": { "predicted_cases": 5, "risk_score": 65.5, "threshold": 0.0, "status": "Medium" },
    "Shivamogga": { "predicted_cases": 11, "risk_score": 64.7, "threshold": 12.0, "status": "Medium" },
    "Mandya": { "predicted_cases": 9, "risk_score": 61.2, "threshold": 3.0, "status": "Medium" },
    "Tumakuru": { "predicted_cases": 4, "risk_score": 61.1, "threshold": 3.25, "status": "Medium" },
    "Raichur": { "predicted_cases": 2, "risk_score": 59.3, "threshold": 0.0, "status": "Medium" },
    "Chikkamagaluru": { "predicted_cases": 5, "risk_score": 54.9, "threshold": 3.0, "status": "Medium" },
    "Vijayapura": { "predicted_cases": 6, "risk_score": 54.1, "threshold": 6.0, "status": "Medium" },
    "Uttara Kannada": { "predicted_cases": 5, "risk_score": 52.2, "threshold": 0.0, "status": "Medium" },
    "Dharwad": { "predicted_cases": 4, "risk_score": 51.6, "threshold": 0.25, "status": "Medium" },
    "Koppal": { "predicted_cases": 4, "risk_score": 49.8, "threshold": 5.0, "status": "Low" },
    "Bagalkote": { "predicted_cases": 4, "risk_score": 49.8, "threshold": 3.0, "status": "Low" },
    "Bidar": { "predicted_cases": 5, "risk_score": 46.3, "threshold": 1.0, "status": "Low" },
    "Kalaburagi": { "predicted_cases": 9, "risk_score": 45.0, "threshold": 6.0, "status": "Low" },
    "Ballari": { "predicted_cases": 4, "risk_score": 43.9, "threshold": 5.25, "status": "Low" },
    "Gadag": { "predicted_cases": 4, "risk_score": 42.8, "threshold": 5.0, "status": "Low" },
    "Hassan": { "predicted_cases": 3, "risk_score": 42.6, "threshold": 3.25, "status": "Low" },
    "Davangere": { "predicted_cases": 4, "risk_score": 41.4, "threshold": 7.0, "status": "Low" },
    "Haveri": { "predicted_cases": 2, "risk_score": 38.6, "threshold": 6.0, "status": "Low" },
    "Dakshina Kannada": { "predicted_cases": 13, "risk_score": 36.7, "threshold": 10.0, "status": "Low" },
    "Bengaluru Urban": { "predicted_cases": 9, "risk_score": 30.7, "threshold": 1.0, "status": "Low" },
    "Udupi": { "predicted_cases": 6, "risk_score": 25.4, "threshold": 8.0, "status": "Low" }
};

// Utility to match GeoJSON names with our data names
const districtNameMap = {
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
    "Chamarajanagar": "Chamarajanagara",
    "Davangere": "Davangere",
    "Gadag": "Gadag",
    "Haveri": "Haveri",
    "Uttara Kannada": "Uttara Kannada",
    "Dakshina Kannada": "Dakshina Kannada",
    "Udupi": "Udupi",
    "Kodagu": "Kodagu",
    "Kolar": "Kolar",
    "Mandya": "Mandya",
    "Tumkur": "Tumakuru"
};

function normalizeName(name) {
    return districtNameMap[name] || name;
}
