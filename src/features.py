# src/features.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop Loan_ID if present
    if "Loan_ID" in df.columns:
        df.drop("Loan_ID", axis=1, inplace=True)

    # Replace target variable values
    if "Loan_Approved" in df.columns:
        df["Loan_Approved"] = df["Loan_Approved"].replace({"Y": 1, "N": 0})

    # Type casting
    df["Credit_History"] = df["Credit_History"].astype("object")
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("object")

    # Impute categorical variables
    df["Gender"].fillna("Male", inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

    # Impute numeric variable
    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df
