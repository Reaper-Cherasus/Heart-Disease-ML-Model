"""
train.py - Train and evaluate heart disease models. Saves the best one.
"""

import os
import sys
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import load_and_clean, prepare_data, FEATURES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "src", "model")
DATA_DIR  = os.path.join(BASE_DIR, "data")

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
}


def train_all(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    results = {}
    print("\n=== Model Results ===")

    for name, model in MODELS.items():
        # Logistic Regression needs scaled data
        Xtr = X_train_scaled if name == "Logistic Regression" else X_train
        Xte = X_test_scaled  if name == "Logistic Regression" else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {"accuracy": acc, "auc": auc, "model": model}

        print(f"\n{name}: Accuracy={acc:.4f} | AUC={auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

    return results


def save_best(results, scaler):
    best_name = max(results, key=lambda x: results[x]["auc"])
    best = results[best_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best["model"], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    meta = {
        "best_model": best_name,
        "accuracy": round(best["accuracy"], 4),
        "auc": round(best["auc"], 4),
        "features": FEATURES,
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest Model: {best_name} | Accuracy: {best['accuracy']:.4f} | AUC: {best['auc']:.4f}")
    print("Saved to model/best_model.pkl")


def plot_feature_importance(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name, color in zip(axes, ["Random Forest", "XGBoost"], ["#2ecc71", "#e74c3c"]):
        imp = pd.Series(results[name]["model"].feature_importances_, index=FEATURES).sort_values(ascending=True)
        imp.plot(kind="barh", ax=ax, color=color, edgecolor="black")
        ax.set_title(f"{name} - Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Feature importance plot saved.")


if __name__ == "__main__":
    print("Loading data...")
    df = load_and_clean()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)

    print("Training models...")
    results = train_all(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

    save_best(results, scaler)
    plot_feature_importance(results)
