import traceback
import shap
import requests
import json


try:
    import joblib
    import pandas as pd

    model = joblib.load("model/churn_model.pkl")
    feature_names = joblib.load("model/feature_names.pkl")

    print("Model loaded successfully!")
    print(f"Features expected: {feature_names}")

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
    churn_prediction = model.predict(test_customer)[0]

    print(f"Churn probability: {churn_probability:.2%}")
    print(f"Will churn: {'Yes' if churn_prediction == 1 else 'No'}")

except Exception as e:
    traceback.print_exc()


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test_customer)

# Get SHAP values for churn class (class 1)
feature_shap = dict(zip(feature_names, shap_values[0][0]))

# Sort by absolute importance
top_factors = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

print("\nTop 3 reasons for this prediction:")
for feature, value in top_factors:
    direction = "increases" if value > 0 else "decreases"
    print(f"  - {feature}: {direction} churn risk (score: {value:.3f})")





# Build the prompt using churn results and SHAP reasons
reasons_text = "\n".join([
    f"- {feature}: {'increases' if value > 0 else 'decreases'} churn risk"
    for feature, value in top_factors
])

prompt = f"""A bank customer has a {churn_probability:.0%} probability of churning.

Top reasons from the model:
{reasons_text}

In 2-3 sentences, explain why this customer might be at risk and suggest one retention action. Keep it simple and business-friendly."""

# Send to Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False
    }
)

llm_explanation = response.json()["response"]
print("\nLLM Explanation:")
print(llm_explanation)