import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Load the dataset and preprocess it:
    1. Handle missing values.
    2. Scale numerical features.
    3. Encode categorical labels.
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    
    # Handle missing values (if any)
    if df.isnull().sum().any():
        print("Handling missing values...")
        df.fillna(df.median(), inplace=True)
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Encode categorical labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    print("Data preprocessing completed!")
    return df, scaler, label_encoder