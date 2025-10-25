import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 🌦️ Weather Prediction App
# -----------------------------
st.set_page_config(page_title="Weather Prediction App", page_icon="🌦️", layout="centered")

st.title("🌤️ Weather Prediction using Ridge Regression")
st.markdown("""
This app predicts **tomorrow's maximum temperature** using a trained Ridge Regression model.  
It accepts raw weather data (`precip`, `temp_max`, `temp_min`) and automatically performs the same feature engineering used during model training.
""")

# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("weather_prediction_model.joblib")
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    st.stop()

# -----------------------------
# Collect User Input
# -----------------------------
st.sidebar.header("Input Today's Weather Data")

precip = st.sidebar.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, value=10.0)
temp_max = st.sidebar.number_input("Today's Max Temperature (°C)", min_value=-10.0, max_value=50.0, value=30.0)
temp_min = st.sidebar.number_input("Today's Min Temperature (°C)", min_value=-10.0, max_value=40.0, value=20.0)

# Combine raw input
raw_input = pd.DataFrame({
    'precip': [precip],
    'temp_max': [temp_max],
    'temp_min': [temp_min]
})

st.subheader("Raw Input Summary")
st.dataframe(raw_input, use_container_width=True)

# -----------------------------
# Feature Engineering (same as training)
# -----------------------------
def engineer_features(df):
    df = df.copy()

    # Use today's max as proxy for 'month_max' (since user only inputs daily data)
    df['month_max'] = df['temp_max']

    # Derived features from your training logic
    df['month_day_max'] = df['month_max'] / df['temp_max']
    df['max_min'] = df['temp_max'] / df['temp_min']

    # Final predictors (same as in training)
    predictors = ['precip', 'temp_max', 'temp_min', 'month_max', 'month_day_max', 'max_min']
    return df[predictors]

engineered_input = engineer_features(raw_input)

st.subheader("Engineered Features (used by the model)")
st.dataframe(engineered_input, use_container_width=True)

# -----------------------------
# Make Prediction
# -----------------------------
if st.button("🔮 Predict Tomorrow's Max Temperature"):
    try:
        prediction = model.predict(engineered_input)[0]
        st.success(f"Predicted Max Temperature for Tomorrow: **{prediction:.2f}°C**")
        st.caption("Model: Trained Ridge Regression Model with Feature Engineering")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("""
**Note:**  
This app automatically performs the same feature engineering as used during training:
- `month_day_max = month_max / temp_max`  
- `max_min = temp_max / temp_min`  
- `month_max` is approximated from today's `temp_max`
""")