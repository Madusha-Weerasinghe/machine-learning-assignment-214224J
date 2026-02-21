import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

INPUT_PATH = "data/processed/processed_wood_apple.csv"
MODEL_PATH = "models/wood_apple_model.pkl"
FEATURE_PATH = "models/wood_apple_features.pkl"

def train():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print("Run preprocessing first!")
        return

    df = pd.read_csv(INPUT_PATH)

    X = df.drop(columns=['Price_F'])
    y = df['Price_F']

    # 🔥 SAVE FEATURE ORDER (IMPORTANT FIX)
    joblib.dump(X.columns.tolist(), FEATURE_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    estimators = [
        ('hgbr', HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
    ]

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV()
    )

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # Evaluation
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    print("\nTRAIN METRICS")
    print("R2:", r2_score(y_train, train_preds))

    print("\nTEST METRICS")
    print("R2:", r2_score(y_test, test_preds))
    print("MAE:", mean_absolute_error(y_test, test_preds))
    print("MSE:", mean_squared_error(y_test, test_preds))

    joblib.dump(model, MODEL_PATH)

    print("✅ Model and feature order saved successfully")


if __name__ == "__main__":
    train()