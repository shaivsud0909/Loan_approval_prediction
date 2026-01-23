import pandas as pd
import joblib
import os
from lime.lime_tabular import LimeTabularExplainer

# ---------------- PATH SETUP ---------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "best_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "loan.csv")

# ---------------- LOAD MODEL ---------------- #

model = joblib.load(model_path)

# ---------------- LOAD DATA ---------------- #

df = pd.read_csv(data_path)

X = df.drop("Loan_Status", axis=1)

# ---------------- PREPROCESS DATA ---------------- #

X_processed = model.named_steps["preprocessing"].transform(X)

# ---------------- CREATE LIME EXPLAINER ---------------- #

explainer = LimeTabularExplainer(

    training_data=X_processed,
    feature_names=model.named_steps["preprocessing"].get_feature_names_out(),
    class_names=["Rejected", "Approved"],
    mode="classification"
)

# ---------------- EXPLAIN SINGLE SAMPLE ---------------- #

sample_index = 0   # change index to explain another row

exp = explainer.explain_instance(

    X_processed[sample_index],
    model.named_steps["classifier"].predict_proba,
    num_features=10
)

# ---------------- SAVE EXPLANATION ---------------- #

output_file = os.path.join(BASE_DIR, "explainability", "lime_explanation.html")

exp.save_to_file(output_file)

print("LIME explanation saved successfully!")
print("File location:", output_file)
