import pandas as pd
from src.data_preprocessing import (
    semantic_inconsistency_masks, prepare_xy
)

def test_phone_semantic_rule():
    df = pd.DataFrame({
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No phone service", "No phone service"],
        "InternetService": ["No", "No"],
        "OnlineSecurity": ["No internet service"] * 2,
        "OnlineBackup": ["No internet service"] * 2,
        "DeviceProtection": ["No internet service"] * 2,
        "TechSupport": ["No internet service"] * 2,
        "StreamingTV": ["No internet service"] * 2,
        "StreamingMovies": ["No internet service"] * 2,
    })
    masks = semantic_inconsistency_masks(df)
    assert masks["MultipleLines: PhoneService=Yes + No phone service"].sum() == 1

def test_prepare_xy_maps_target(sample_df):
    X, y = prepare_xy(sample_df)
    assert set(y.unique()) <= {0, 1}
    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
