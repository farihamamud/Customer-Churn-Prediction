import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from load_data import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df = load_data()
y = df["Churn"]
X = pd.get_dummies(df.drop(columns="Churn"), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("BASELINE (always predicts 'stays')")
print(classification_report(y_test, baseline.predict(X_test), zero_division=0))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("LOGISTIC REGRESSION")
print(classification_report(y_test, model.predict(X_test)))



balanced = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
balanced.fit(X_train, y_train)
print("LOGISTIC REGRESSION (balanced)")
print(classification_report(y_test, balanced.predict(X_test)))

forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
print("RANDOM FOREST")
print(classification_report(y_test, forest.predict(X_test)))



pred = balanced.predict(X_test)
print(confusion_matrix(y_test, pred))

missed = X_test[(pred == 0) & (y_test.values == 1)]
caught = X_test[(pred == 1) & (y_test.values == 1)]
print("Missed churners, average tenure:", missed["tenure"].mean())
print("Caught churners, average tenure:", caught["tenure"].mean())

missed_rows = df.loc[missed.index]
print(missed_rows["Contract"].value_counts())
print(df[df["Churn"] == 1]["Contract"].value_counts())