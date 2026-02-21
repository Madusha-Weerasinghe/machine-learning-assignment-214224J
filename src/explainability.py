import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score

# 1. Configuration & Paths
DATA_PATH = 'data/processed/processed_wood_apple.csv'
OUTPUT_DIR = 'output'
TARGET_COL = 'Price_F'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 2. Load Data and Train Model
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found. Please run preprocessing first.")
    exit()

df = pd.read_csv(DATA_PATH)
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Training the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X, y)
preds = model.predict(X)
r2 = r2_score(y, preds)

# --- DIAGRAM 1: ACCURACY & FEATURE DRIVERS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Clean Accuracy Scatter Plot
ax1.scatter(y, preds, alpha=0.5, color='#6a4c93', edgecolors='w', label='Sampled Data')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
ax1.set_title(f"Wood Apple Accuracy (Overall R²={r2:.2f})", fontsize=14)
ax1.set_xlabel("Actual Price (Rs.)")
ax1.set_ylabel("Predicted Price (Rs.)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Feature Importance Bar Chart
importances = model.feature_importances_
indices = np.argsort(importances)
ax2.barh(range(len(indices)), importances[indices], color='#4eb851')
ax2.set_yticks(range(len(indices)))
ax2.set_yticklabels([X.columns[i] for i in indices])
ax2.set_title("Wood Apple Price Drivers", fontsize=14)
ax2.set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wood_apple_accuracy_importance.png'), dpi=300)
plt.close()

# --- DIAGRAM 2: SENSITIVITY DYNAMICS (SMOOTHED) ---
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Wood Apple Price Dynamics Analysis (Overall R²={r2:.2f})", fontsize=18)

# Features based on your specific processed CSV columns
features = ['region', 'Humid', 'Rain (mm)', 'Temp_C']
colors = ['#6a4c93', '#4eb851', '#2d5a27', '#e76f51']

for i, feature in enumerate(features):
    if feature in X.columns:
        ax = axs[i//2, i%2]
        
        # Sort data and apply rolling mean for the "Clean" look
        sorted_df = pd.DataFrame({feature: X[feature], 'pred': preds}).sort_values(by=feature)
        window_size = max(1, len(X)//10)
        sorted_df['smoothed'] = sorted_df['pred'].rolling(window=window_size, min_periods=1, center=True).mean()
        
        ax.plot(sorted_df[feature], sorted_df['smoothed'], color=colors[i], lw=3)
        ax.set_title(f"Impact of {feature}", fontsize=13)
        ax.set_ylabel("Predicted Price (Rs.)")
        ax.set_xlabel(feature)
        ax.grid(True, linestyle='--', alpha=0.5)
    else:
        print(f"Warning: Feature '{feature}' not found in data.")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'wood_apple_sensitivity_dynamics.png'), dpi=300)
plt.close()

print(f"✅ Success! Generated accuracy and sensitivity diagrams in '{OUTPUT_DIR}'")