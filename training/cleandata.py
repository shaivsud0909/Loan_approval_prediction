import pandas as pd

def clean_data(df):

    df.drop("Loan_ID", axis=1, inplace=True)

    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)

    
    # ---------------- FEATURE ENGINEERING ---------------- #

    # Total Income Feature
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Loan to Income Ratio (Avoid division by zero)
    df["LoanIncomeRatio"] = df["LoanAmount"] / (df["TotalIncome"] + 1)

    return df
