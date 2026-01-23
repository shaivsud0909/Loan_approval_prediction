from api.schema import LoanInput
import pandas as pd
import joblib
import os
import shap
import numpy as np

# ---------------- PATH SETUP ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "best_model.pkl")

# ---------------- LOAD MODEL ---------------- #

model = joblib.load(model_path)

# ---------------- SHAP EXPLAINER ---------------- #

explainer = shap.Explainer(model.named_steps["classifier"])


# ---------------- PREDICTION SERVICE ---------------- #

def prediction(features: LoanInput):

    # Convert input to dataframe
    input_data = features.dict()
    df = pd.DataFrame([input_data])

    # ---------------- FEATURE ENGINEERING  ---------------- #

    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Avoid division by zero
    df["LoanIncomeRatio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)

    # ---------------- PREPROCESS INPUT ---------------- #

    X_processed = model.named_steps["preprocessing"].transform(df)

    # ---------------- MODEL OUTPUT ---------------- #

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    # ---------------- SHAP EXPLANATION ---------------- #

    shap_values = explainer(X_processed)

    feature_names = model.named_steps["preprocessing"].get_feature_names_out()

    # Extract Approved class SHAP values
    shap_row = shap_values.values[0][:, 1]

    # Select top 5 important features
    top_idx = np.argsort(np.abs(shap_row))[-5:][::-1]

    explanation = []

    for i in top_idx:
        explanation.append({
            "feature": feature_names[i],
            "impact": round(float(shap_row[i]), 4)
        })

    # ---------------- RESPONSE ---------------- #

    result = {
        "loan_status": "Approved" if pred == "Y" else "Rejected",
        "approval_probability": round(float(prob), 3),
        "top_factors": explanation
    }

    return result
