import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import LabelEncoder
import joblib

RAW_PATH = "data/raw/prices.csv"
PROCESSED_PATH = "data/processed/processed_wood_apple.csv"
ENCODER_PATH = "data/processed/region_encoder.pkl"

def preprocess():
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(RAW_PATH):
        print(f"{RAW_PATH} not found!")
        return

    df = pd.read_csv(RAW_PATH, low_memory=False)

    # =========================================================
    # 1️⃣ Filter ONLY Wood Apple
    # =========================================================
    df = df[df['Fruit_Type'].str.contains("wood apple", case=False, na=False)].copy()
    print(f"Total Wood Apple records: {len(df)}")

    # =========================================================
    # 2️⃣ Date Processing
    # =========================================================
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    df['month'] = df['Date'].dt.month
    df['dayofweek'] = df['Date'].dt.dayofweek

    # =========================================================
    # 3️⃣ Clean Numeric Columns
    # =========================================================
    def clean_val(val):
        if pd.isna(val):
            return np.nan
        res = re.search(r"(\d+\.?\d*)", str(val))
        return float(res.group(1)) if res else np.nan

    for col in ['Temp_C', 'Rain (mm)', 'Humid']:
        df[col] = df[col].apply(clean_val)

    # =========================================================
    # 4️⃣ Interaction Feature (Heat Index)
    # =========================================================
    df['heat_index'] = df['Temp_C'] * df['Humid']

    # =========================================================
    # 5️⃣ Remove Price Outliers (5%–95%)
    # =========================================================
    low, high = df['Price_F'].quantile([0.05, 0.95])
    df = df[(df['Price_F'] >= low) & (df['Price_F'] <= high)].copy()

    # =========================================================
    # 6️⃣ STANDARDIZE REGION NAMES (🔥 IMPORTANT FIX)
    # =========================================================
    df['region'] = (
        df['region']
        .astype(str)
        .str.strip()          # remove leading/trailing spaces
        .str.lower()          # convert to lowercase
        .str.replace(r"\s+", " ", regex=True)  # remove double spaces
        .str.title()          # convert to clean Title Case
    )

    print("Unique Regions After Cleaning:")
    print(sorted(df['region'].unique()))

    # =========================================================
    # 7️⃣ Drop Unnecessary Columns
    # =========================================================
    cols_to_drop = ['Date', 'Fruit_Type', 'Veg_Type', 'Price_V', 'Score']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # =========================================================
    # 8️⃣ Fill Missing Values
    # =========================================================
    df = df.fillna(df.median(numeric_only=True))

    # =========================================================
    # 9️⃣ Encode Region
    # =========================================================
    le = LabelEncoder()
    df['region'] = le.fit_transform(df['region'])

    joblib.dump(le, ENCODER_PATH)

    # =========================================================
    # 🔟 Save Processed Dataset
    # =========================================================
    df.to_csv(PROCESSED_PATH, index=False)

    print("\n✅ Preprocessing Complete")
    print("Final Features:", df.columns.tolist())


if __name__ == "__main__":
    preprocess()