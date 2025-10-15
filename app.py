import streamlit as st
import numpy as np
import joblib
import time

# Load model
model = joblib.load("wine.pkl")  # <-- your saved model file

# Page config
st.set_page_config(page_title="Wine Quality Predictor 🍷", page_icon="🍇", layout="centered")

# Title and intro
st.title("🍷 Wine Quality Prediction App")
st.write("### Predict the **quality of wine** from its chemical composition 🍇")
st.markdown("#### Adjust the inputs from the sidebar and click **Predict** to see the result.")

# Sidebar inputs
st.sidebar.header("🧪 Input Wine Features")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, 7.0)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, 0.076)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.sidebar.number_input("Density", 0.0, 2.0, 0.9978)
pH = st.sidebar.number_input("pH", 0.0, 5.0, 3.51)
sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.sidebar.number_input("Alcohol", 0.0, 20.0, 9.4)

# Prepare input
input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])

# Prediction section
if st.button("🔮 Predict Wine Quality"):
    with st.spinner("Analyzing wine sample... 🍷"):
        time.sleep(1.5)  # simulate computation delay
        prediction = model.predict(input_data)
        pred_value = prediction[0]
    
    st.balloons()  # 🎈 Fun animation after prediction
    
    # Show result dynamically
    if isinstance(pred_value, (int, float, np.integer, np.floating)):
        st.success(f"### 🧾 Predicted Wine Quality Score: **{pred_value:.2f}** ⭐")
        if pred_value >= 7:
            st.markdown("### 🍇 Excellent Wine! Premium Quality!")
            st.snow()
        elif 5 <= pred_value < 7:
            st.markdown("### 🍷 Good Quality Wine — Enjoy responsibly!")
        else:
            st.markdown("### ⚠️ Average or Low Quality Wine — Could be improved.")
    else:
        st.success(f"### 🍇 Predicted Wine Category: **{pred_value}**")

# Footer
st.write("---")
st.markdown("Developed with ❤️ using **Streamlit** and **Joblib**")
