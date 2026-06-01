import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="HomeValue AI", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    model  = joblib.load('linear_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

st.title("🏠 HomeValue AI")
st.subheader("House Price Predictor")
st.write("Built on Linear Regression trained on 1,460 real houses")
st.divider()

sqft     = st.slider("Living Area (sqft)", 500,  5000, 1500, 50)
bedrooms = st.slider("Bedrooms",             1,     8,    3,  1)
fullbath = st.slider("Full Bathrooms",       1,     4,    2,  1)
halfbath = st.slider("Half Bathrooms",       0,     2,    1,  1)

st.divider()

if st.button("✦ Calculate Price", use_container_width=True):
    features = np.array([[sqft, bedrooms, fullbath, halfbath]])
    features_scaled = scaler.transform(features)
    price = model.predict(features_scaled)[0]

    st.success(f"### Estimated Price: ${price:,.0f}")
    st.info(f"Price Range: ${price*0.92:,.0f} — ${price*1.08:,.0f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Base Price",  "$180,502")
    col2.metric("Living Area", f"+${53440.16 * sqft/1000:,.0f}")
    col3.metric("Bedrooms",    f"${-21777.09 * bedrooms:,.0f}")
    col4.metric("Bathrooms",   f"+${16988.85 * fullbath + 2318.34 * halfbath:,.0f}")

st.divider()
st.caption("Trained on Ames Housing Dataset · SkillCraft Internship Task 01")
