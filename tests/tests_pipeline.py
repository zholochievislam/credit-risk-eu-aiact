import pytest
import pandas as pd
import numpy as np
from src.data_processing import clean_raw_data, get_preprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'person_age': [23, 134, 30, 23],  # outlier 144 and duplicate
        'person_income': [50000, 60000, 70000, 50000],
        'person_emp_length': [2.0, np.nan, 5.0, 2.0],  # missing value
        'loan_amnt': [10000, 5000, 12000, 10000],
        'loan_int_rate': [10.5, 12.0, np.nan, 10.5],  # missing value
        'loan_percent_income': [0.2, 0.1, 0.15, 0.2],
        'cb_person_cred_hist_length': [2, 10, 4, 2],
        'person_home_ownership': ['RENT', 'OWN', 'RENT', 'RENT'],
        'loan_intent': ['PERSONAL', 'EDUCATION', 'MEDICAL', 'PERSONAL'],
        'loan_grade': ['A', 'B', 'B', 'A'],
        'cb_person_default_on_file': ['N', 'N', 'Y', 'N'],
        'loan_status': [0, 1, 0, 0]
    })

def test_clean_raw_data(sample_data):
    cleaned = clean_raw_data(sample_data)
    assert len(cleaned) == 3
    assert cleaned['person_age'].max() <= 100
    assert cleaned['person_emp_length'].dropna().max() <= 60

def test_preprocessor(sample_data):
    cleaned = clean_raw_data(sample_data)
    num_cols = ['person_age', 'person_income', 'person_emp_length','loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    preprocessor = get_preprocessor(num_cols, cat_cols)
    X_trans = preprocessor.fit_transform(cleaned[num_cols+cat_cols])

    assert np.isnan(X_trans).sum() == 0



