import joblib
import pandas as pd

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("churn_model.pkl")

# Extract expected features from pipeline safely
MODEL_FEATURES = list(model.feature_names_in_)

# -----------------------------
# Threshold (can later move to config)
# -----------------------------
THRESHOLD = 0.5

# -----------------------------
# SAFE FEATURE EXTRACTION
# -----------------------------
try:
    MODEL_FEATURES = list(model.feature_names_in_)
except:
    # fallback for pipelines
    MODEL_FEATURES = model.steps[0][1].feature_names_in_
# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_input(input_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data])

    # ADD missing columns
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # REMOVE extra columns (VERY IMPORTANT FIX)
    df = df[MODEL_FEATURES]

    # enforce exact order
    df = df.reindex(columns=MODEL_FEATURES)

    return df

# -----------------------------
# Prediction function
# -----------------------------
THRESHOLD = 0.5

def predict_churn(input_data: dict):
    df = preprocess_input(input_data)

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= THRESHOLD)

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": pred
    }


# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":

    sample_customer = {
        "CreditScore": 600,
        "Age": 40,
        "Tenure": 3,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 70000,
        "Point Earned": 400,

        "Geography_Germany": 0,
        "Geography_Spain": 1,

        "Gender_Male": 1,

        "Card Type_GOLD": 0,
        "Card Type_PLATINUM": 0,
        "Card Type_SILVER": 1
    }

    result = predict_churn(sample_customer)

    print("🔥 Churn Prediction Result:")
    print(result)