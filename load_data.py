import pandas as pd

def load_data(path="Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    # New customers (tenure 0) have not been billed yet, so total charges = 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df.drop(columns=["customerID"])]
    