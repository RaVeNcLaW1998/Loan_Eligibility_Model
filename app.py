import os
import streamlit as st
import pandas as pd

from src.config import RAW_DATA_PATH
from src.dataset import load_data, split_data
from src.features import preprocess_data
from src.modeling.train import train_model
from src.modeling.predict import load_model, make_prediction

MODEL_PATH = "models/logistic_model.pkl"

st.title("🏦 Loan Eligibility Predictor")

# Step 1: Train the model on first run if model not found
if not os.path.exists(MODEL_PATH):
    st.info("Training model on credit.csv for the first time...")
    df = pd.read_csv(RAW_DATA_PATH)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    train_model(X_train, y_train)
    st.success("Model trained and saved.")

# Step 2: Input form
st.header("Enter Applicant Details")
with st.form("loan_form"):
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.selectbox(
        "Loan Term", ["360.0", "180.0", "120.0", "84.0", "60.0", "36.0", "12.0"]
    )
    Credit_History = st.selectbox("Credit History", ["1.0", "0.0"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("Check Eligibility")

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "Gender": Gender,
                    "Married": Married,
                    "Dependents": Dependents,
                    "Education": Education,
                    "Self_Employed": Self_Employed,
                    "ApplicantIncome": ApplicantIncome,
                    "CoapplicantIncome": CoapplicantIncome,
                    "LoanAmount": LoanAmount,
                    "Loan_Amount_Term": Loan_Amount_Term,
                    "Credit_History": Credit_History,
                    "Property_Area": Property_Area,
                }
            ]
        )

        preprocessed_input = preprocess_data(input_data)

        model = load_model(MODEL_PATH)
        prediction = make_prediction(model, preprocessed_input)

        st.subheader("Prediction Result")
        st.success("✅ Loan Approved" if prediction == 1 else "❌ Loan Denied")
