import pandas as pd

df = pd.read_csv("Telco-Customer-Churn.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df["Churn"].value_counts())