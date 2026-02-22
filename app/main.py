import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Paths for assets
MODEL_PATH = "models/wood_apple_model.pkl"
ENCODER_PATH = "data/processed/region_encoder.pkl"
FEATURE_PATH = "models/wood_apple_features.pkl"
ACCURACY_CHART = "output/wood_apple_accuracy_importance.png"
SENSITIVITY_CHART = "output/wood_apple_sensitivity_dynamics.png"

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
        color: #000000 !important; /* Forces text color to black */
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

    # ---------- TAB 1: PREDICTION ----------
    with tab1:
        if not os.path.exists(MODEL_PATH):
            st.error("Model not found. Run training first.")
        else:
            model = joblib.load(MODEL_PATH)
            encoder = joblib.load(ENCODER_PATH)
            feature_order = joblib.load(FEATURE_PATH)

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
                        <p>Current Predicted Market Price</p>
                    </div>
                """, unsafe_allow_html=True)

                st.divider()

                # --- LOCAL EXPLANATION LOGIC ---
                st.subheader("🕵️ Why is the price Rs. {:.2f}?".format(prediction))
                
                baseline_data = input_data.copy()
                baseline_data['Temp_C'] = 28.0
                baseline_data['Humid'] = 70.0
                baseline_data['Rain (mm)'] = 10.0
                baseline_data['heat_index'] = 28.0 * 70.0
                base_p = model.predict(baseline_data)[0]

                exp_cols = st.columns(3)
                
                h_data = baseline_data.copy()
                h_data['Humid'] = humid
                h_data['heat_index'] = h_data['Temp_C'] * humid
                h_impact = model.predict(h_data)[0] - base_p
                exp_cols[0].metric("Humidity Impact", f"Rs. {h_impact:+.2f}", f"{humid}% vs 70% avg")

                t_data = baseline_data.copy()
                t_data['Temp_C'] = temp
                t_data['heat_index'] = temp * t_data['Humid']
                t_impact = model.predict(t_data)[0] - base_p
                exp_cols[1].metric("Temperature Impact", f"Rs. {t_impact:+.2f}", f"{temp}°C vs 28°C avg")

                r_data = baseline_data.copy()
                r_data['Rain (mm)'] = rain
                r_impact = model.predict(r_data)[0] - base_p
                exp_cols[2].metric("Rainfall Impact", f"Rs. {r_impact:+.2f}", f"{rain}mm vs 10mm avg")

                # The reasoning box with black text color applied via CSS
                st.markdown(f"""
                <div class="reasoning-box">
                <strong>Model Logic:</strong> The price is primarily driven by 
                {'higher than average' if h_impact > 0 else 'lower than average'} humidity and 
                {'warmer' if t_impact > 0 else 'cooler'} temperatures in <strong>{region}</strong>.
                Combined, these factors created a net adjustment of <strong>Rs. {h_impact + t_impact + r_impact:+.2f}</strong> relative to standard baseline conditions.
                </div>
                """, unsafe_allow_html=True)

    # ---------- TAB 2: XAI ANALYTICS ----------
    with tab2:
        st.header("Global Model Intelligence")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(ACCURACY_CHART):
                st.image(ACCURACY_CHART, caption="Feature Importance Breakdown")
        with c2:
            if os.path.exists(SENSITIVITY_CHART):
                st.image(SENSITIVITY_CHART, caption="Global Trend Analysis")

   # ---------- TAB 3: ABOUT PROJECT ----------
    with tab3:
        st.header("📖 Project Documentation & Market Context")
        
        st.subheader("⚠️ The Problem")
        st.markdown("""
        Investing in Wood Apple products or raw fruit procurement in Sri Lanka often faces several challenges:
        * **Price Volatility:** Market prices fluctuate unpredictably due to seasonal changes and monsoon patterns.
        * **Information Asymmetry:** Small-scale investors and manufacturers often lack access to real-time data on how environmental factors impact costs.
        * **Investment Risk:** Without data-driven insights, businesses may overpay for stock or fail to anticipate supply shortages during extreme weather.
        """)

        st.subheader("💡 The Solution")
        st.markdown("""
        This **Wood Apple Price Intelligence** system provides a data-driven solution to these challenges:
        * **Informed Investment:** Investors can use the prediction tool to estimate future costs and decide the best time to purchase or process products.
        * **Risk Mitigation:** By analyzing the **Local Baseline**, users can understand exactly why a price is high or low (e.g., due to high humidity or regional scarcity).
        * **Transparent Logic:** Unlike "black-box" models, this system uses **XAI (Explainable AI)** to show how parameters like Rainfall and Temperature drive the final market price.
        """)

        st.subheader("🛠 Technical Methodology")
        st.markdown("""
        * **Model:** A sophisticated **Stacking Regressor** that combines the strengths of *HistGradientBoosting* and *Random Forest* models.
        * **Explainability:** Utilizing **Local Baseline Analysis** to show individual parameter impacts compared to average Sri Lankan environmental conditions.
        * **Tech Stack:** Built with Python, Streamlit, Scikit-Learn, and XGBoost.
        """)
if __name__ == "__main__":
    main()