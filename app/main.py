import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

# Paths for assets
MODEL_PATH = "models/wood_apple_model.pkl"
ENCODER_PATH = "data/processed/region_encoder.pkl"
FEATURE_PATH = "models/wood_apple_features.pkl"
DATA_PATH = "data/processed/processed_wood_apple.csv"

# Evaluation Chart Paths
CHART_ACTUAL_VS_PRED = "output/actual_vs_predict_50.png"
CHART_ACCURACY_SCATTER = "output/prediction_accuracy_scatter.png"
CHART_IMPORTANCE = "output/feature_importance.png"
CHART_SHAP_SUMMARY = "output/wood_apple_accuracy_importance.png"

st.set_page_config(
    page_title="Wood Apple AI Dashboard",
    page_icon="🍏",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .main-title { font-size: 34px; font-weight: 700; color: #2E7D32; }
    .sub-title { font-size: 18px; color: gray; }
    .metric-card {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    }
    .reasoning-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-left: 5px solid #2E7D32;
        border-radius: 4px;
        margin: 10px 0;
        color: #000000 !important;
        font-size: 18px;
        font-weight: 500;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-title">🍏 Wood Apple Price Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Agricultural Market Prediction System</div>', unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["🚀 Price Prediction", "📊 XAI Analytics", "ℹ️ About Project"])

    # ---------- TAB 1: PREDICTION & INDIVIDUAL IMPACT ----------
    with tab1:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
            st.error("Required files (model or data) not found. Run training first.")
        else:
            model = joblib.load(MODEL_PATH)
            encoder = joblib.load(ENCODER_PATH)
            feature_order = joblib.load(FEATURE_PATH)
            
            # Load a small fixed reference for consistent SHAP base values
            df_ref = pd.read_csv(DATA_PATH).drop(columns=['Price_F'])
            background_sample = shap.sample(df_ref[feature_order], 100)

            st.sidebar.header("📊 Input Parameters")
            temp = st.sidebar.number_input("Temperature (°C)", value=30.0)
            humid = st.sidebar.number_input("Humidity (%)", value=75.0)
            rain = st.sidebar.number_input("Rainfall (mm)", value=15.0)
            region = st.sidebar.selectbox("Market Region", list(encoder.classes_))
            month = st.sidebar.slider("Month", 1, 12, 6)
            day = st.sidebar.slider("Day (0=Mon, 6=Sun)", 0, 6, 2)
            predict_btn = st.sidebar.button("🚀 Predict Price")

            if predict_btn:
                heat_index = temp * humid
                encoded_region = encoder.transform([region])[0]
                
                input_data = pd.DataFrame({
                    'Temp_C': [temp], 'Rain (mm)': [rain], 'Humid': [humid],
                    'month': [month], 'dayofweek': [day], 'heat_index': [heat_index],
                    'region': [encoded_region]
                })[feature_order]

                prediction = model.predict(input_data)[0]

                st.markdown(f"""
                    <div class="metric-card">
                        <h1 style="color:#1B5E20;">Rs. {prediction:.2f}</h1>
                        <p>Predicted Market Price</p>
                    </div>
                """, unsafe_allow_html=True)

                st.divider()

                # --- CONSISTENT INDIVIDUAL IMPACT ANALYSIS ---
                st.subheader(f"🕵️ Why is the price Rs. {prediction:.2f}?")
                
                # Use the fixed background sample so the "starting bar" (E[f(x)]) never changes
                explainer = shap.Explainer(model.predict, background_sample)
                shap_values = explainer(input_data)

                fig, ax = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)

                st.markdown(f"""
                <div class="reasoning-box">
                <strong>How to interpret this:</strong> This chart starts from the global average price (bottom). 
                <span style="color:red;">Red bars</span> show your specific inputs that increased the price, 
                while <span style="color:blue;">blue bars</span> show inputs that decreased it.
                </div>
                """, unsafe_allow_html=True)

    # ---------- TAB 2: XAI ANALYTICS ----------
    with tab2:
        st.header("Global Model Performance & Intelligence")
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            if os.path.exists(CHART_ACTUAL_VS_PRED):
                st.image(CHART_ACTUAL_VS_PRED, caption="Actual vs. Predicted (First 50 Samples)")
        with row1_col2:
            if os.path.exists(CHART_ACCURACY_SCATTER):
                st.image(CHART_ACCURACY_SCATTER, caption="Overall Prediction Accuracy")

        st.divider()

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            if os.path.exists(CHART_IMPORTANCE):
                st.image(CHART_IMPORTANCE, caption="XGBoost Feature Importance")
        with row2_col2:
            if os.path.exists(CHART_SHAP_SUMMARY):
                st.image(CHART_SHAP_SUMMARY, caption="SHAP Global Summary")

    # ---------- TAB 3: ABOUT PROJECT ----------
    with tab3:
        st.header("📖 Project Documentation & Market Context")
        
        # Section 1: The Problem
        st.subheader("⚠️ The Problem: Investment Risk & Information Gaps")
        st.markdown("""
        Investing in Wood Apple products or raw fruit procurement in Sri Lanka faces significant hurdles:
        * **Price Volatility:** Agricultural commodities like Wood Apple experience high price instability due to seasonal changes and monsoon patterns.
        * **Information Asymmetry:** Small-scale producers and investors often lack access to real-time data, leading to financial risk during procurement.
        * **Perishability:** High post-harvest losses in Sri Lanka impact market value; understanding environmental drivers is crucial for minimizing waste.
        """)

        # Section 2: The Solution
        st.subheader("💡 The Solution: Data-Driven Price Intelligence")
        st.markdown("""
        This system provides a technical framework to mitigate these risks:
        * **Risk Mitigation:** By analyzing individual feature impacts, users can understand if a price is driven by temporary stressors (like high humidity) or regional scarcity.
        * **Transparent Logic:** This system uses **XAI (Explainable AI)** to show how parameters like Rainfall and Temperature drive the final market price.
        * **Investment Optimization:** Investors can estimate costs based on current conditions to decide the optimal time for fruit processing or procurement.
        """)

        # Section 3: Technical Methodology
        st.subheader("🛠 Technical Methodology & Data Processing")
        
        col_tech1, col_tech2 = st.columns(2)
        with col_tech1:
            st.markdown("""
            **Data Pipeline:**
            * **Feature Engineering:** Raw market data is integrated with environmental metrics, including a custom **Heat Index** (Temp × Humid) to capture combined stress factors.
            * **Model Selection:** We utilize a high-performance **XGBoost Regressor** (500 estimators, max depth 6) optimized for tabular agricultural data.
            """)
        
        with col_tech2:
            st.markdown("""
            **Explainability & Validation:**
            * **SHAP Integration:** The system uses **SHAP (SHapley Additive Explanations)** to mathematically attribute price changes to specific features.
            * **Rigorous Evaluation:** The model is validated using R2 Score, Correlation, MAE, and MSE to ensure reliability.
            """)

            # Display the evaluation metrics from the generated file
        if os.path.exists("output/model_evaluation.txt"):
            with st.expander("📊 View Latest Model Evaluation Report"):
                with open("output/model_evaluation.txt", "r") as f:
                    st.text(f.read())

if __name__ == "__main__":
    main()