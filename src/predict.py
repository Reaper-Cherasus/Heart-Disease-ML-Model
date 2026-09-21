"""
predict.py - Make predictions using the trained model.
"""

import os
import json
import joblib
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "src", "model")

# Feature columns (must match training order)
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    return model, scaler


def load_model_meta():
    path = os.path.join(MODEL_DIR, "model_meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def predict(patient: dict) -> dict:
    model, scaler = load_model()
    meta = load_model_meta()

    # Always use a DataFrame to keep feature names
    df = pd.DataFrame([patient])[FEATURES]

    # Only scale for Logistic Regression
    if meta.get("best_model") == "Logistic Regression":
        X = pd.DataFrame(scaler.transform(df), columns=FEATURES)
    else:
        X = df

    prediction = int(model.predict(X)[0])
    probability = round(float(model.predict_proba(X)[0][1]), 4)

    return {
        "prediction": prediction,
        "probability": probability,
        "label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "risk_level": "High" if probability >= 0.6 else "Moderate" if probability >= 0.3 else "Low",
        "model_used": meta.get("best_model", "Unknown"),
    }


if __name__ == "__main__":
    sample = {
        "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 165, "exang": 0,
        "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 2
    }
    result = predict(sample)
    print("Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
