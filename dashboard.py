import streamlit as st
import pandas as pd
import numpy as np
import joblib

#  PAGE CONFIG 
st.set_page_config(page_title="TEP AI Diagnosis", page_icon="🏭", layout="wide")

# 1. LOAD THE BRAIN
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('tep_xgboost_model.pkl')
        scaler = joblib.load('tep_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# 2. SIDEBAR
st.sidebar.title("Control Room")
st.sidebar.info("Status: System Online")

# 3. MAIN INTERFACE
st.title("Chemical Process Fault Diagnosis")
st.markdown("Upload raw sensor data to detect Reactor & Separator faults.")

if model is None:
    st.error(" ERROR: .pkl files not found. Make sure they are in the same folder!")
    st.stop()

#  4. DATA UPLOAD 
uploaded_file = st.file_uploader("Upload Sensor CSV (52 Columns)", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        
        # 5. FEATURE ENGINEERING
        # A. Calculate Velocity
        df_velocity = df_input.iloc[:, :52].diff().fillna(0)
        
        # B. Combine
        X_combined = pd.concat([df_input.iloc[:, :52], df_velocity], axis=1)
        
        # C. Scale
        X_scaled = scaler.transform(X_combined)
        
        # D. Predict (Shift 0-19 back to 1-20)
        preds_human = model.predict(X_scaled) + 1
        
        # 6. RESULTS
        st.subheader("Diagnostic Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latest Fault Detected", f"Fault {preds_human[-1]}")
        with col2:
            st.metric("Rows Processed", len(df_input))

        st.line_chart(preds_human)
        
        df_results = df_input.copy()
        df_results['AI_Diagnosis'] = preds_human
        st.dataframe(df_results)

    except Exception as e:
        st.error(f"Processing Error: {e}")
