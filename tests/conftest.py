import pandas as pd
import pytest
from src.data_preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

@pytest.fixture
def sample_df():
    row = {c: 0 for c in NUMERIC_FEATURES}
    row.update({c: "No" for c in CATEGORICAL_FEATURES})
    row.update({
        "customerID": "A",
        "Churn": "Yes",
        "gender": "Male",
        "InternetService": "No",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
    })
    return pd.DataFrame([row])
