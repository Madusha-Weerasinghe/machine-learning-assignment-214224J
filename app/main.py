import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/wood_apple_model.pkl"
ENCODER_PATH = "data/processed/region_encoder.pkl"
FEATURE_PATH = "models/wood_apple_features.pkl"

st.set_page_config(
    page_title="Wood Apple AI Dashboard",
    page_icon="🍏",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #2E7D32;
    }
    .sub-title {
        font-size: 18px;
        color: gray;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


def main():

    # ---------- HEADER ----------
    st.markdown('<div class="main-title">🍏 Wood Apple Price Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Agricultural Market Prediction System</div>', unsafe_allow_html=True)
    st.divider()

    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run training first.")
        return

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_order = joblib.load(FEATURE_PATH)

    # ---------- SIDEBAR ----------
    st.sidebar.header("📊 Input Parameters")

    temp = st.sidebar.number_input("Temperature (°C)", value=30.0)
    humid = st.sidebar.number_input("Humidity (%)", value=75.0)
    rain = st.sidebar.number_input("Rainfall (mm)", value=15.0)

    region_list = list(encoder.classes_)
    region = st.sidebar.selectbox("Market Region", region_list)

    month = st.sidebar.slider("Month", 1, 12, 6)
    day = st.sidebar.slider("Day (0=Mon, 6=Sun)", 0, 6, 2)

    predict_btn = st.sidebar.button("🚀 Predict Price")

    # ---------- MAIN DASHBOARD ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"🌡 Temperature\n\n{temp} °C")

    with col2:
        st.info(f"💧 Humidity\n\n{humid} %")

    with col3:
        st.info(f"🌧 Rainfall\n\n{rain} mm")

    st.divider()

    if predict_btn:

        heat_index = temp * humid
        encoded_region = encoder.transform([region])[0]

        input_data = pd.DataFrame({
            'Temp_C': [temp],
            'Rain (mm)': [rain],
            'Humid': [humid],
            'month': [month],
            'dayofweek': [day],
            'heat_index': [heat_index],
            'region': [encoded_region]
        })

        input_data = input_data[feature_order]

        prediction = model.predict(input_data)[0]

        st.markdown("## 💰 Predicted Market Price")
        st.markdown(
            f"""
            <div class="metric-card">
                <h1 style="color:#1B5E20;">Rs. {prediction:.2f}</h1>
                <p>Estimated Wood Apple Market Price</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # Extra Info Section
        st.markdown("### 📈 Market Context")
        st.write(f"""
        • Region: **{region}**  
        • Month: **{month}**  
        • Day of Week: **{day}**  
        • Heat Index: **{heat_index:.2f}**
        """)


if __name__ == "__main__":
    main()