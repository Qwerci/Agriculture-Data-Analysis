import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_data(df, soil_moisture_data):
    """
    Create visualizations for soil moisture trends and rainfall data.
    """
    print("Generating visualizations...")
    
    # Set the style for seaborn plots
    sns.set(style="whitegrid")
    
    # 1. Soil Moisture Trend
    dates = pd.to_datetime(soil_moisture_data['date'])
    moisture = soil_moisture_data['moisture']
    df_moisture = pd.DataFrame({'date': dates, 'moisture': moisture}).set_index('date')
    df_moisture = df_moisture.resample('D').mean().ffill()  # Handle missing data
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_moisture.index, df_moisture['moisture'], marker='o', color='blue')
    plt.title('Soil Moisture Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Soil Moisture (%)')
    plt.grid()
    plt.show()
    
    # 2. Rainfall Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df['rainfall'], bins=30, kde=True, color='green')
    plt.title('Distribution of Rainfall')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Frequency')
    plt.show()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()