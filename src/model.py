import joblib
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(df):
    """
    Train a RandomForestClassifier on the dataset.
    """
    print("Training model...")
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    return model

# Save and load the model
def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Load a trained model from a file.
    """
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model