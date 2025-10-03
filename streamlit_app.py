
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model pipeline
bundle = joblib.load("model.joblib")
model = bundle["model"]
features = bundle["features"]

st.set_page_config(page_title="📊 Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Predict which customers are likely to leave and explore key insights.")

# Sidebar choice
mode = st.sidebar.radio("Select Mode:", ["🔹 Single Prediction", "📂 Bulk CSV Upload"])

# -----------------------------
# Mode 1: Single Prediction
# -----------------------------
if mode == "🔹 Single Prediction":
    st.sidebar.header("🔍 Enter Customer Details")

    def user_input_features():
        data = {
            "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
            "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
            "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
            "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
            "tenure": st.sidebar.slider("Tenure (months)", 0, 72, 12),
            "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
            "InternetService": st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
            "Contract": st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
            "PaymentMethod": st.sidebar.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ]),
            "MonthlyCharges": st.sidebar.number_input("Monthly Charges", 10, 150, 70),
            "TotalCharges": st.sidebar.number_input("Total Charges", 0, 10000, 1000)
        }
        return pd.DataFrame([data])

    input_df = user_input_features()
# -----------------------------
# Mode 2: Bulk CSV Upload
# -----------------------------
elif mode == "📂 Bulk CSV Upload":
    st.subheader("📂 Upload a CSV File for Churn Prediction")

    uploaded_file = st.file_uploader("Drag and drop your CSV here", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("✅ Uploaded Data Preview:")
        st.dataframe(data.head())

        # Ensure only required features are used
        missing_cols = [col for col in features if col not in data.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in CSV: {missing_cols}")
        else:
            preds = model.predict_proba(data[features])[:, 1]
            data["Churn_Probability"] = preds
            data["Prediction"] = ["🚨 Churn" if p > 0.5 else "✅ Stay" for p in preds]

            st.subheader("🔮 Prediction Results")
            st.dataframe(data[["customerID", "Churn_Probability", "Prediction"]]
                         if "customerID" in data.columns else data.head())

            # Download results
            csv_out = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv_out, "churn_predictions.csv", "text/csv")

# -----------------------------
# Business Insights Section
# -----------------------------
st.subheader("📈 Business Insights")
st.markdown("""
- Customers with **Month-to-Month contracts** and **Electronic check payments** have higher churn risk.  
- Higher **Monthly Charges** + **Short Tenure** often signal churn.  
- Offering **discounts, loyalty rewards, or longer contracts** can reduce churn.
""")

st.subheader("📊 Feature Importance")
st.image("feature_importance.png")
