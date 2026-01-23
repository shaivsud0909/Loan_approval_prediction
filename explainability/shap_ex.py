import shap
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build correct paths
model_path = os.path.join(BASE_DIR, "best_model.pkl")
data_path = os.path.join(BASE_DIR, "data", "loan.csv")

# Load model
model = joblib.load(model_path)

# Load data
df = pd.read_csv(data_path)

# Drop target
X = df.drop("Loan_Status", axis=1)

#explainer 
explainer = shap.Explainer(model.named_steps["classifier"])

# Transform data using preprocessing
X_processed = model.named_steps["preprocessing"].transform(X)

# Calculate SHAP values
shap_values = explainer(X_processed)

# Global feature importance plot
shap.summary_plot(shap_values, X_processed, show=False)

print("SHAP global explanation generated")
