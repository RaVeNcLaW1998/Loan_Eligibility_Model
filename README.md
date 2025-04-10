# 🏦 Loan Eligibility Prediction App (Streamlit)

This project is a machine learning web app built using **Streamlit** that predicts whether a loan should be approved or denied based on user-input features.

---

## 🚀 Project Features

- Trains a logistic regression model on `credit.csv` (stored in `data/raw/`)
- Automatically handles missing values, encoding, and type conversions
- Uses consistent preprocessing for both training and prediction
- Accepts user input via Streamlit form interface
- Predicts loan eligibility and displays the result

---

## 🧠 Model

- **Model Type**: Logistic Regression
- **Trained On**: Preprocessed version of `credit.csv`
- **Features**:
  - Gender, Married, Dependents, Education, Self_Employed
  - ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
  - Credit_History, Property_Area

---

## 📁 Project Structure

```
Loan_Eligibility_Model/
│
├── data/
│   └── raw/credit.csv          # Original dataset
│
├── models/
│   └── logistic_model.pkl      # Trained model saved here
│
├── src/                        # Modular Python scripts
│   ├── config.py               # Constants
│   ├── dataset.py              # Data loading and splitting
│   ├── features.py             # Preprocessing (imputation, encoding)
│   └── modeling/
│       ├── train.py            # Training logic
│       └── predict.py          # Prediction logic
│
├── streamlit_app.py            # Streamlit UI
└── README.md                   # You're here!
```

---

## ✅ Setup Instructions

### 1. Clone the Project

```bash
git clone <your-repo-url>
cd Loan_Eligibility_Model
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run streamlit_app.py
```

---

## 📌 Notes

- If `models/logistic_model.pkl` does not exist, the app will automatically train a model using `data/raw/credit.csv`.
- Preprocessing steps (like dropping `Loan_ID`, encoding, and imputing missing values) are reused for both training and prediction via `preprocess_data()` in `features.py`.

---

## 📬 Contact

Created by Athul Krishna Radhakrishnan Nair. For questions or suggestions, open an issue or contact me at [athulkrishnar1998@gmail.com]
