import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🧠 DDoS Attack Detection using KNN")
st.write("Upload a CSV file to predict whether the network traffic is normal or a DDoS attack.")

# Load model and feature list
try:
    model = joblib.load("knn_model.pkl")
    feature_names = joblib.load("feature_names_knn.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or feature names: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("✅ File uploaded successfully!")
        st.write(df.head())

        # Strip column names of extra whitespace
        df.columns = df.columns.str.strip()

        # Check if all expected features exist
        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            st.error(f"⚠️ Your CSV is missing the following required columns:\n\n{missing_features}")
        else:
            # Use only necessary columns, handle any missing values
            X_new = df[feature_names].fillna(0)

            # Make predictions
            predictions = model.predict(X_new)

            # Show results
            df["Prediction"] = predictions

            # Normalize prediction values to string
            df["Prediction"] = df["Prediction"].astype(str).str.strip().str.upper()

            # Map known labels
            label_map = {
                "0": "Normal",
                "1": "DDoS",
                "BENIGN": "Normal",
                "DDOS": '🚨 DDoS Attack Detected'
            }
            df["Prediction_Label"] = df["Prediction"].map(label_map).fillna("Unknown")


            st.success("✅ Predictions completed!")
            st.write(df[["Prediction", "Prediction_Label"]])

            # Offer to download results
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
else:
    st.info("📁 Please upload a file to begin.")
