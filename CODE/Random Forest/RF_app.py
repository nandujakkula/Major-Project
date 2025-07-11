import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import joblib

st.set_page_config(page_title="DDoS Attack Classifier", layout="wide")
def add_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://img.freepik.com/free-vector/stream-binary-code-design-vector_53876-170628.jpg?uid=R163070998&ga=GA1.1.631535332.1729694839&semt=ais_hybrid&w=740");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
add_bg_image("")


# Load the model and encoder
model = joblib.load("ddos_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


st.markdown("""
    <style>
    h1, p {
        color: white;
        text-align: left;
        font-size: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown(
    '<h1 style="color: grey; text-align: center">🚨 Detecting DDoS Attack Classification Web App</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="color: white; font-size: 20px">Upload a CSV file to detect DDoS attack types using a trained RandomForest model.</p>',
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # 🚨 Clean up column names
        data.columns = data.columns.str.strip()
        model_columns = [col.strip() for col in model.feature_names_in_]

        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(data.head())

        # Check for missing columns
        missing_cols = [col for col in model_columns if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            st.success("✅ All required columns found. Running predictions...")

            # Select only the required columns
            input_data = data[model_columns]

            predictions = model.predict(input_data)
            prediction_probs = model.predict_proba(input_data)

            decoded_predictions = label_encoder.inverse_transform(predictions)
            prob_df = pd.DataFrame(prediction_probs, columns=label_encoder.classes_)
            result_df = data.copy()
            result_df["Predicted_Label"] = decoded_predictions

            # Count the number of each prediction label
            label_counts = result_df["Predicted_Label"].value_counts()
            
            # Pie chart
            st.subheader("🧩 Distribution of Predicted Labels")
            plt.figure(figsize=(1, 6))  # Set smaller figure size
            fig, ax = plt.subplots(figsize=(7, 7))  # Set smaller figure size
            ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis("equal")  # Equal aspect ratio ensures pie chart is circular.
            st.pyplot(fig)



            st.subheader("🧾 Prediction Results")
            st.dataframe(result_df[["Predicted_Label"]])

            st.subheader("📈 Prediction Probabilities")
            st.dataframe(prob_df)

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")
else:
    st.info("Please upload a CSV file to begin.")



# Add this to your Streamlit app to print required columns
st.write("✅ Expected Columns for Prediction:")
st.code(list(model.feature_names_in_))

#"C:\Users\jakku\OneDrive\Desktop\animation.lottie"

# streamlit run app.py to run the application

