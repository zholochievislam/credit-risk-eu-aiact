import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()

    age_median = df.loc[df['person_age'] <= 100, 'person_age'].median()
    emp_median = df.loc[df['person_emp_length'] <= 60, 'person_emp_length'].median()

    df.loc[df['person_age'] > 100, 'person_age'] = age_median
    df.loc[df['person_emp_length'] > 60, 'person_emp_length'] = emp_median
    return df

def get_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output = False))])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    return preprocessor

if __name__ == "__main__":
    filepath = "/Users/islam/Desktop/Budapest_Bootcamp/data/raw/credit_risk_dataset.csv"
    try:
        raw_data = pd.read_csv(filepath)
        cleaned_data = clean_raw_data(raw_data)

        num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length']
        cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

        X = cleaned_data[num_cols+cat_cols]
        y = cleaned_data[['loan_status']]

        preprocessor = get_preprocessor(num_cols, cat_cols)
        X_processed = preprocessor.fit_transform(X)
        print("___"*15)
        print(f"Raw Data Shape:       {raw_data.shape}")
        print("___" * 15)
        print(f"Cleaned Data Shape:   {cleaned_data.shape}")
        print("___" * 15)
        print(f"Processed Array Shape:{X_processed.shape}")
        print("___" * 15)
        print(f"Missing values left:  {np.isnan(X_processed).sum()}")

    except FileNotFoundError:
        print(f"File not found")







