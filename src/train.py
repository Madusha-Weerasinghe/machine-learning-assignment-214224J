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
REPORT_PATH = "output/model_evaluation.txt"

def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print("Run preprocessing first!")
        return

    df = pd.read_csv(INPUT_PATH)
    X = df.drop(columns=['Price_F'])
    y = df['Price_F']

    joblib.dump(X.columns.tolist(), FEATURE_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    estimators = [
        ('hgbr', HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42))
    ]
    model = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # Predictions for both sets
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Metrics calculation
    train_r2 = r2_score(y_train, train_preds)
    train_corr = np.corrcoef(y_train, train_preds)[0, 1]

    test_r2 = r2_score(y_test, test_preds)
    test_corr = np.corrcoef(y_test, test_preds)[0, 1]
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)

    # Structured Evaluation Report
    metrics_report = f"""
    WOOD APPLE MODEL EVALUATION REPORT
    =================================
    TRAINING SET METRICS
    - R2 Score: {train_r2:.4f}
    - Correlation: {train_corr:.4f}

    TEST SET METRICS
    - R2 Score: {test_r2:.4f}
    - Correlation: {test_corr:.4f}
    - Mean Absolute Error (MAE): {test_mae:.2f}
    - Mean Squared Error (MSE): {test_mse:.2f}
    """

    with open(REPORT_PATH, "w") as f:
        f.write(metrics_report)

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Training Complete. Separate metrics saved to {REPORT_PATH}")

if __name__ == "__main__":
    train()