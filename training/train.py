import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from cleandata import clean_data
from hyper import tune_random_forest

df=pd.read_csv("data/loan.csv")
df=clean_data(df)

numerical_cols=df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols=df.select_dtypes(include=['object']).columns.tolist()


X=df.drop(["Loan_Status"],axis=1)
y=df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=y
)


categorical_cols.remove("Loan_Status")

preprocessor=ColumnTransformer([
    ('num',StandardScaler(),numerical_cols),
    ('cat',OneHotEncoder(handle_unknown="ignore"),categorical_cols)
])

model_pipeline = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier())
    ]
)

search = tune_random_forest(model_pipeline, X_train, y_train)

best_model = search.best_estimator_  #newly trained pipeline with best hyperparameter

print("Best Parameters:")
print(search.best_params_)

# Training performance
y_train_pred = best_model.predict(X_train)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred) * 100)
print(confusion_matrix(y_train, y_train_pred))


# Test performance
y_test_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred) * 100)
print(confusion_matrix(y_test, y_test_pred))


import joblib

joblib.dump(best_model, "best_model.pkl")
