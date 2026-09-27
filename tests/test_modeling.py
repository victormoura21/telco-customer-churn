import numpy as np
from src.modeling import evaluate_predictions

def test_evaluate_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = evaluate_predictions(y, p, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0
