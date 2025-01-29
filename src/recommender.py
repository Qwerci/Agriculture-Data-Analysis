
from analyse import analyze_moisture_trend

CROP_PARAMS = {
    "rice": {"min_moisture": 60, "max_moisture": 80, "min_rainfall": 200},
    "wheat": {"min_moisture": 45, "max_moisture": 60, "min_rainfall": 150},
    "corn": {"min_moisture": 50, "max_moisture": 70, "min_rainfall": 180},
    "soybean": {"min_moisture": 40, "max_moisture": 55, "min_rainfall": 120}
}

def recommend_irrigation(model, scaler, label_encoder, current_data, soil_moisture_data):
    """
    Generate a farmer-friendly irrigation recommendation using the trained model and crop-specific thresholds.
    """
    # Predict the crop type
    current_scaled = scaler.transform([list(current_data.values())])
    predicted_crop = label_encoder.inverse_transform(model.predict(current_scaled))[0]
    
    # Analyze soil moisture trend
    trend = analyze_moisture_trend(soil_moisture_data)
    
    # Get crop-specific thresholds
    crop_rules = CROP_PARAMS.get(predicted_crop, {"min_moisture": 40, "max_moisture": 60, "min_rainfall": 150})
    
    # Current soil moisture and rainfall
    current_moisture = current_data['K']  # Using Potassium (K) as a proxy for soil moisture
    current_rainfall = current_data['rainfall']
    
    # Generate farmer-friendly recommendation
    recommendation = []
    if current_moisture < crop_rules["min_moisture"]:
        recommendation.append(f"🚨 **Urgent Action Required**: Soil moisture is critically low ({current_moisture}%). "
                             f"Irrigate immediately to avoid crop stress.")
    elif current_moisture < (crop_rules["min_moisture"] + 5) and trend == "decreasing":
        recommendation.append(f"⚠️ **Advisory**: Soil moisture is approaching the minimum threshold ({current_moisture}%) "
                             f"and is trending downward. Consider irrigating soon.")
    
    if current_rainfall < crop_rules["min_rainfall"]:
        recommendation.append(f"⚠️ **Advisory**: Rainfall is insufficient ({current_rainfall}mm). "
                             f"Your crop requires at least {crop_rules['min_rainfall']}mm of rainfall per week. "
                             f"Consider supplemental irrigation.")
    
    if not recommendation:
        recommendation.append("✅ **No Action Needed**: Soil moisture and rainfall are within optimal ranges. "
                             "Your crop is in good condition.")
    
    # Compile insights
    insights = {
        "Predicted Crop": predicted_crop,
        "Current Soil Moisture": f"{current_moisture}%",
        "Optimal Soil Moisture Range": f"{crop_rules['min_moisture']}%-{crop_rules['max_moisture']}%",
        "Current Rainfall": f"{current_rainfall}mm",
        "Required Rainfall": f"{crop_rules['min_rainfall']}mm",
        "Soil Moisture Trend": trend
    }
    
    return recommendation, insights