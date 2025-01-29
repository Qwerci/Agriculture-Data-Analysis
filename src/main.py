from cleaning import load_and_preprocess_data
from model import load_model, save_model, train_model
from recommender import recommend_irrigation


def main():
    # Load and preprocess data
    file_path = '../data/Crop_recommendation.csv' 
    df, scaler, label_encoder = load_and_preprocess_data(file_path)
    
    # Train and save the model
    model = train_model(df)
    save_model(model, 'crop_recommendation_model.pkl')
    
    # Load the model (for demonstration)
    model = load_model('crop_recommendation_model.pkl')
    
    # Example input from farmer
    current_data = {
        'N': 85, 'P': 45, 'K': 40,  # Soil nutrients
        'temperature': 25, 'humidity': 65, 'ph': 6.5, 'rainfall': 120  # Weather data
    }
    
    # Example soil moisture data (last 7 days)
    soil_moisture_data = {
        'date': ['2023-10-01', '2023-10-02', '2023-10-03', '2023-10-04', '2023-10-05'],
        'moisture': [50, 48, 45, 43, 40]  # Moisture readings (%)
    }
    
    # Generate irrigation recommendation
    recommendation, insights = recommend_irrigation(model, scaler, label_encoder, current_data, soil_moisture_data)
    
    # Output results
    print("\n🌱 **Irrigation Recommendation** 🌱")
    for rec in recommendation:
        print(f"- {rec}")
    
    print("\n📊 **Insights**")
    for key, value in insights.items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    main()