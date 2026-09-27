from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

INTERNET_DEPENDENT_FEATURES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

EXPECTED_COLUMNS = (
    [ID_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
)


def load_data(path):
    """Carrega a planilha e aplica apenas coerções técnicas seguras."""
    df = pd.read_excel(Path(path))

    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    # A planilha fornecida já vem numérica, mas a coerção deixa o pipeline robusto.
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_data(df):
    """Validações que não alteram semanticamente a base."""
    if df.empty:
        raise ValueError("Dataset vazio.")

    invalid_target = set(df[TARGET].dropna().unique()) - {"Yes", "No"}
    if invalid_target:
        raise ValueError(f"Valores inesperados em Churn: {invalid_target}")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Há customerID duplicado.")

    return True


def semantic_inconsistency_masks(df):
    """
    Regras de consistência entre serviço-base e serviços dependentes.
    Servem para auditoria; não corrigimos automaticamente o baseline.
    """
    masks = {}

    masks["MultipleLines: PhoneService=Yes + No phone service"] = (
        df["PhoneService"].eq("Yes")
        & df["MultipleLines"].eq("No phone service")
    )
    masks["MultipleLines: PhoneService=No + Yes/No"] = (
        df["PhoneService"].eq("No")
        & df["MultipleLines"].isin(["Yes", "No"])
    )

    has_internet = ~df["InternetService"].eq("No")
    no_internet = df["InternetService"].eq("No")

    for col in INTERNET_DEPENDENT_FEATURES:
        masks[f"{col}: has internet + No internet service"] = (
            has_internet & df[col].eq("No internet service")
        )
        masks[f"{col}: InternetService=No + Yes/No"] = (
            no_internet & df[col].isin(["Yes", "No"])
        )

    return masks


def audit_semantic_inconsistencies(df):
    masks = semantic_inconsistency_masks(df)
    rows = []
    any_issue = pd.Series(False, index=df.index)

    for rule, mask in masks.items():
        any_issue |= mask
        rows.append({
            "rule": rule,
            "count": int(mask.sum()),
            "pct_rows": float(mask.mean()),
        })

    table = pd.DataFrame(rows).sort_values("count", ascending=False)
    summary = {
        "rows": int(len(df)),
        "total_rule_violations": int(table["count"].sum()),
        "affected_customers": int(any_issue.sum()),
        "affected_pct": float(any_issue.mean()),
    }
    return table.reset_index(drop=True), summary


def prepare_xy(df):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Target contém valores ausentes ou inesperados.")
    return X, y.astype(int)


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
