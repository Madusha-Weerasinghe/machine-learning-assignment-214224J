import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

INPUT_PATH = "data/processed/processed_wood_apple.csv"
MODEL_PATH = "models/wood_apple_model.pkl"
FEATURE_PATH = "models/wood_apple_features.pkl"
REPORT_PATH = "output/model_evaluation.txt"
OUTPUT_DIR = "output"

def generate_visualizations(model, X, y):
    """Generates performance and explainability charts."""
    print("📊 Generating model visualizations...")
    preds = model.predict(X)
    
    # 1. Actual vs Predicted for first 50 samples
    plt.figure(figsize=(12, 6))
    plt.plot(range(50), y[:50], 'o-', label='Actual Price', color='#1B5E20')
    plt.plot(range(50), preds[:50], 's--', label='Predicted Price', color='#4eb851')
    plt.title("Actual vs Predicted Wood Apple Prices (First 50 Samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Price (Rs.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predict_50.png'), bbox_inches='tight')
    plt.close()

    # 2. Prediction Accuracy Scatter Plot (All Data)
    plt.figure(figsize=(8, 8))
    plt.scatter(y, preds, alpha=0.5, color='#2E7D32')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title("Prediction Accuracy: Actual vs Predicted (All Data)")
    plt.xlabel("Actual Price (Rs.)")
    plt.ylabel("Predicted Price (Rs.)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_accuracy_scatter.png'), bbox_inches='tight')
    plt.close()

    # 3. XGBoost Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='weight', ax=plt.gca(), color='#4eb851')
    plt.title("XGBoost Feature Importance (Weight)")
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), bbox_inches='tight')
    plt.close()

    # 4. SHAP Summary Plot (Workaround for base_score error)
    background_data = shap.sample(X, 100)
    explainer = shap.Explainer(model.predict, background_data)
    shap_values = explainer(X)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary: Feature Impact on Price")
    plt.savefig(os.path.join(OUTPUT_DIR, 'wood_apple_accuracy_importance.png'), bbox_inches='tight')
    plt.close()

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

    print("🚀 Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)

    # Predictions for metrics
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Metrics calculation
    train_r2 = r2_score(y_train, train_preds)
    train_corr = np.corrcoef(y_train, train_preds)[0, 1]

    test_r2 = r2_score(y_test, test_preds)
    test_corr = np.corrcoef(y_test, test_preds)[0, 1]
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)

    
    metrics_report = f"""
    WOOD APPLE MODEL EVALUATION REPORT (XGBOOST)
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
    
    # Generate charts
    generate_visualizations(model, X, y)
    
    print(f"✅ Training Complete. Metrics and all charts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()