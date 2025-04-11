import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💼 Loan Default Prediction App")
st.write("Enter loan application details:")

# Input fields
loan_amount = st.number_input("Loan Amount (₹)", value=30000)
income = st.number_input("Monthly Income (₹)", value=60000)
loan_term = st.number_input("Loan Term (months)", value=24)
interest_rate = st.number_input("Interest Rate (%)", value=11.5)
employment_length = st.number_input("Employment Length (years)", value=5)
credit_score = st.number_input("Credit Score", value=650)
payment_history_text = st.selectbox("Payment History", ['Poor', 'Average', 'Good', 'Excellent'])

# Encode payment history
payment_map = {'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}
payment_history = payment_map[payment_history_text]

# Feature Engineering
loan_to_income_ratio = loan_amount / income if income != 0 else 0
interest_to_income_ratio = (interest_rate * loan_amount) / income if income != 0 else 0
is_long_term = 1 if loan_term > 36 else 0
loan_burden_score = loan_to_income_ratio + interest_to_income_ratio

# Prediction
if st.button("Predict Default Risk"):
    input_data = np.array([
        loan_amount, income, loan_term, interest_rate, employment_length,
        credit_score, payment_history, loan_to_income_ratio,
        interest_to_income_ratio, is_long_term, loan_burden_score
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    risk_score = model.predict_proba(input_scaled)[0][1]
    prediction = 1 if risk_score >= 0.5 else 0

    st.markdown(f"### 🧮 Risk Score: `{round(risk_score, 2)}`")
    st.markdown(f"### 🔍 Prediction: `{'Defaulter' if prediction == 1 else 'Non-Defaulter'}`")

    if prediction == 1:
        if risk_score > 0.7:
            st.markdown("### 💡 Recommendation: ❌ **Reject** (High Risk)")
        elif risk_score > 0.4:
            st.markdown("### 💡 Recommendation: ⚠️ **Manual Review**")
        else:
            st.markdown("### 💡 Recommendation: ✅ **Approve with Caution**")
    else:
        st.markdown("### 💡 Recommendation: ✅ **Approve**")