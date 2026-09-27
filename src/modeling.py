from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
)
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_data, validate_data, prepare_xy, build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_candidates():
    """Modelos e grades compactas para uma busca reproduzível e eficiente."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200],
                "model__max_depth": [5, None],
                "model__min_samples_leaf": [2, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_iter": [150],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
    }

def evaluate_predictions(y_true, proba, threshold=0.50):
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def tune_on_training(X_train, y_train):
    """
    Toda escolha de algoritmo/hiperparâmetro ocorre apenas no treino.
    O holdout não participa desta função.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

    rows = []
    searches = {}

    for name, (estimator, grid) in make_candidates().items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])

        search = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches[name] = search

        idx = search.best_index_
        rows.append({
            "model": name,
            "cv_roc_auc_mean": search.cv_results_["mean_test_roc_auc"][idx],
            "cv_roc_auc_std": search.cv_results_["std_test_roc_auc"][idx],
            "cv_pr_auc_mean": search.cv_results_["mean_test_pr_auc"][idx],
            "cv_pr_auc_std": search.cv_results_["std_test_pr_auc"][idx],
            "best_params": json.dumps(search.best_params_, ensure_ascii=False),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["cv_roc_auc_mean", "cv_pr_auc_mean"],
        ascending=False,
    ).reset_index(drop=True)

    selected_name = summary.loc[0, "model"]
    selected_estimator = searches[selected_name].best_estimator_
    return summary, selected_name, selected_estimator


def oof_business_metrics(estimator, X_train, y_train,
                         capacities=(0.10, 0.20, 0.30, 0.40, 0.50)):
    """
    Previsões OOF do treino: threshold/capacidade não são escolhidos no holdout.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(
        clone(estimator), X_train, y_train,
        cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    temp = pd.DataFrame({
        "y": np.asarray(y_train),
        "proba": proba,
    }).sort_values("proba", ascending=False).reset_index(drop=True)

    prevalence = temp["y"].mean()
    positives = temp["y"].sum()
    rows = []

    for k in capacities:
        n = max(1, int(np.ceil(len(temp) * k)))
        selected = temp.iloc[:n]
        precision = selected["y"].mean()
        recall = selected["y"].sum() / positives
        rows.append({
            "top_k": k,
            "customers_contacted": n,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "lift_at_k": precision / prevalence,
        })

    threshold_rows = []
    for threshold in np.linspace(0.20, 0.80, 61):
        threshold = round(float(threshold), 2)
        m = evaluate_predictions(y_train, proba, threshold)
        m["contact_rate"] = float((proba >= threshold).mean())
        threshold_rows.append(m)

    return (
        pd.DataFrame(rows),
        pd.DataFrame(threshold_rows),
        proba,
    )


def coefficient_table(fitted_pipeline):
    """Interpretação do vencedor quando ele for Logistic Regression."""
    model = fitted_pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        return None

    names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()
    coef = model.coef_[0]

    out = pd.DataFrame({
        "feature": [
            x.replace("numeric__", "").replace("categorical__", "")
            for x in names
        ],
        "coefficient": coef,
        "odds_ratio": np.exp(coef),
    })
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def run_pipeline(data_path, output_dir="reports/generated", model_dir="models"):
    df = load_data(data_path)
    validate_data(df)
    X, y = prepare_xy(df)

    # O holdout é separado ANTES de qualquer seleção supervisionada de modelo.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    summary, selected_name, selected = tune_on_training(X_train, y_train)

    # O vencedor já vem ajustado em todo o treino pelo GridSearchCV.
    test_proba = selected.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_proba, threshold=0.50)

    topk, thresholds, oof_proba = oof_business_metrics(
        selected, X_train, y_train
    )

    coefs = coefficient_table(selected)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    summary.to_csv(out / "model_selection_cv.csv", index=False)
    pd.DataFrame([{
        "selected_model": selected_name, **test_metrics
    }]).to_csv(out / "final_holdout_metrics.csv", index=False)
    topk.to_csv(out / "oof_top_k_metrics.csv", index=False)
    thresholds.to_csv(out / "oof_threshold_curve.csv", index=False)

    if coefs is not None:
        coefs.to_csv(out / "logistic_coefficients.csv", index=False)

    pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "proba": test_proba,
    }).to_csv(out / "holdout_predictions.csv", index=False)

    joblib.dump(selected, Path(model_dir) / "final_model.joblib")

    return {
        "selection": summary,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "top_k": topk,
        "thresholds": thresholds,
        "coefficients": coefs,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

def load_generated_results(output_dir="reports/generated"):
    """
    Carrega os artefatos já gerados pelo pipeline completo.

    Útil para notebooks de análise posterior: evita repetir GridSearchCV
    apenas para visualizar métricas e interpretar o modelo salvo.
    """
    out = Path(output_dir)
    required = {
        "selection": out / "model_selection_cv.csv",
        "test_metrics": out / "final_holdout_metrics.csv",
        "top_k": out / "oof_top_k_metrics.csv",
        "thresholds": out / "oof_threshold_curve.csv",
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Resultados ainda não foram gerados. Rode primeiro o notebook "
            "02_Modeling.ipynb ou `python main.py`. Ausentes: " + ", ".join(missing)
        )

    result = {
        "selection": pd.read_csv(required["selection"]),
        "test_metrics": pd.read_csv(required["test_metrics"]).iloc[0].to_dict(),
        "top_k": pd.read_csv(required["top_k"]),
        "thresholds": pd.read_csv(required["thresholds"]),
    }

    coef_path = out / "logistic_coefficients.csv"
    result["coefficients"] = (
        pd.read_csv(coef_path) if coef_path.exists() else None
    )

    model_path = Path("models/final_model.joblib")
    result["model_path"] = model_path if model_path.exists() else None
    return result

