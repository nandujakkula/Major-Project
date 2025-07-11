import streamlit as st
import pandas as pd
import joblib

st.title("🌐 DDoS Attack Detection using DECISION TREE ")
st.write("Upload a CSV file to classify traffic using a Decision Tree model.")

# Load model and features
model = joblib.load("decision_tree_full.pkl")
feature_names = joblib.load("feature_names_DT.pkl")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            st.error(f"Missing required features: {', '.join(missing)}")
        else:
            input_data = df[feature_names].values
            predictions = model.predict(input_data)

            df['Prediction'] = predictions
            df['Prediction'] = ['✅ Normal Traffic' if pred == 'BENIGN' else '🚨 DDoS Attack Detected' for pred in predictions]

            st.success("✅ Predictions completed!")
            st.subheader("🔍 Prediction Results")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "ddos_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")

else:
    st.info("📁 Please upload a CSV file.")
