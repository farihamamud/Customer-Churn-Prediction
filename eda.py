from load_data import load_data

df = load_data()

print(df.groupby("Churn")["tenure"].mean())
print(df.groupby("Contract")["Churn"].mean())
print(df.groupby("InternetService")["Churn"].mean())
print(df.groupby("PaymentMethod")["Churn"].mean())