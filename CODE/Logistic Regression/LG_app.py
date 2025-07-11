import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Logistic classifier", layout="wide")
st.title("🧠 DDoS Attack Detection using Logistic Regression")
st.write("Upload a CSV file to predict whether the network traffic is Normal or DDoS.")

# Load logistic regression model and feature list
try:
    model = joblib.load("logistic_model.pkl")
    feature_names = joblib.load("feature_names_lg.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or features: {e}")
    st.stop()

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        st.success("✅ File uploaded successfully!")
        st.write("📄 Preview of uploaded data:", df.head())

        # Ensure required features are present
        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            st.error(f"⚠️ Missing required columns: {missing_features}")
        else:
            X_new = df[feature_names].fillna(0)

            # Predict
            predictions = model.predict(X_new)
            df["Prediction"] = predictions

            # Map predictions to labels
            df["Prediction"] = df["Prediction"].astype(str).str.strip().str.upper()
            label_map = {
                "0": "Normal",
                "1": "DDoS",
                "BENIGN": "Normal",
                "DDOS": "🚨 DDoS Attack Detected"
            }
            df["Prediction_Label"] = df["Prediction"].map(label_map).fillna("Unknown")

            st.success("✅ Predictions completed!")
            st.write(df[["Prediction", "Prediction_Label"]])

            # Show prediction counts
            st.write("📊 Prediction Summary:")
            summary = df["Prediction_Label"].value_counts().rename_axis('Label').reset_index(name='Count')
            st.dataframe(summary)

            # 📈 Bar chart
            st.subheader("📊 Prediction Distribution")
            fig, ax = plt.subplots()
            sns.barplot(x="Label", y="Count", data=summary, palette="Set2", ax=ax)
            ax.set_title("Distribution of Predictions")
            st.pyplot(fig)

            # Download option
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", data=csv, file_name="logistic_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📂 Please upload a CSV file to get started.")












