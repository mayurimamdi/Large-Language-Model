import traceback

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
        'Geography_Germany': 0,
        'Geography_Spain': 0,
        'Gender_Male': 1
    }])

    churn_probability = model.predict_proba(test_customer)[0][1]
    churn_prediction = model.predict(test_customer)[0]

    print(f"Churn probability: {churn_probability:.2%}")
    print(f"Will churn: {'Yes' if churn_prediction == 1 else 'No'}")

except Exception as e:
    traceback.print_exc()