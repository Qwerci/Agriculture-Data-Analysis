import pandas as pd
import numpy as np

def analyze_moisture_trend(soil_moisture_data):
    """
    Calculate soil moisture trend over the last 7 days.
    """
    dates = pd.to_datetime(soil_moisture_data['date'])
    moisture = soil_moisture_data['moisture']
    df = pd.DataFrame({'date': dates, 'moisture': moisture}).set_index('date')
    df = df.resample('D').mean().ffill()  # Handle missing data
    trend = np.polyfit(np.arange(len(df)), df['moisture'], 1)[0]
    return "increasing" if trend > 0.5 else "decreasing" if trend < -0.5 else "stable"