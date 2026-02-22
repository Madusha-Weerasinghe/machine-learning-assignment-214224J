import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score

DATA_PATH = 'data/processed/processed_wood_apple.csv'
OUTPUT_DIR = 'output'
TARGET_COL = 'Price_F'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(DATA_PATH)
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X, y)
preds = model.predict(X)

# Chart 1: Key Drivers
plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices], color='#4eb851')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Key Wood Apple Price Drivers")
plt.savefig(os.path.join(OUTPUT_DIR, 'wood_apple_accuracy_importance.png'))

# Chart 2: Trends
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
features = ['region', 'Humid', 'Rain (mm)', 'Temp_C']
for i, feature in enumerate(features):
    ax = axs[i//2, i%2]
    sorted_df = pd.DataFrame({feature: X[feature], 'pred': preds}).sort_values(by=feature)
    sorted_df['smoothed'] = sorted_df['pred'].rolling(window=len(X)//10, center=True).mean()
    ax.plot(sorted_df[feature], sorted_df['smoothed'], color='#2E7D32', lw=3)
    ax.set_title(f"Trend: Price vs {feature}")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wood_apple_sensitivity_dynamics.png'))
print("✅ XAI Charts Generated.")