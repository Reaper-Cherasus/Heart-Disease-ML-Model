"""
data_pipeline.py - Load and prepare the heart disease dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "src", "model")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Binarize target: 0 = no disease, 1+ = disease
    df["target"] = (df["target"] > 0).astype(int)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df


def prepare_data(df):
    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features and save scaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Loaded {len(df)} rows | Target: {df['target'].value_counts().to_dict()}")
    X_train, X_test, *_ = prepare_data(df)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
