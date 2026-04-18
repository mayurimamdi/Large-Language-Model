import streamlit as st
import joblib
import pandas as pd
import shap
import requests

# Load model
model = joblib.load("model/churn_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")

st.title("Bank Churn Chatbot")
st.caption("Ask me about any customer's churn risk")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask about a customer e.g. Will this customer churn?")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Run prediction on test customer
    test_customer = pd.DataFrame([{
        'CreditScore': 650,
        'Age': 45,
        'Tenure': 3,
        'Balance': 0.0,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 0,
        'EstimatedSalary': 50000,
        'Point Earned': 350,
        'Geography_Germany': 0,
        'Geography_Spain': 0,
        'Gender_Male': 1,
        'Card Type_GOLD': 0,
        'Card Type_PLATINUM': 0,
        'Card Type_SILVER': 1
    }])

    churn_probability = model.predict_proba(test_customer)[0][1]

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_customer)
    feature_shap = dict(zip(feature_names, shap_values[0][0]))
    top_factors = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    reasons_text = "\n".join([
        f"- {f}: {'increases' if v > 0 else 'decreases'} churn risk"
        for f, v in top_factors
    ])

    # LLM
    prompt = f"""A bank customer has a {churn_probability:.0%} probability of churning.

Top reasons:
{reasons_text}

The user asked: {user_input}

Answer in 2-3 sentences, business-friendly."""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1:8b", "prompt": prompt, "stream": False}
    )

    answer = response.json()["response"]

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)