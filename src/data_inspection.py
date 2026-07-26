import pandas as pd

df = pd.read_csv("/Users/islam/Desktop/Budapest_Bootcamp/data/raw/credit_risk_dataset.csv")
print("Number of rows: ", len(df))
print("Number of columns: ", len(df.columns))

print("_____"*10)

print("Types:")
print(df.dtypes)

print("_____"*10)

print("Missing values:")
print(df.isnull().sum())

print("_____"*10)

print("Duplicate values:")
print(df.duplicated().sum())

print("_____"*10)

print("Loan Status distribution:")
print(df["loan_status"].value_counts(normalize=True))

print("_____"*10)

print("Person_Age summary:")
print(df["person_age"].describe())