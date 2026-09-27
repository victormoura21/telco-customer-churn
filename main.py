from pathlib import Path

from src.data_preprocessing import (
    load_data, validate_data, audit_semantic_inconsistencies
)
from src.modeling import run_pipeline

DATA_PATH = Path("data/raw/telco_customer_churn.xlsx")


def main():
    df = load_data(DATA_PATH)
    validate_data(df)

    audit, audit_summary = audit_semantic_inconsistencies(df)

    print("\n=== Dataset ===")
    print(f"Linhas: {len(df):,}")
    print(f"Colunas: {df.shape[1]}")
    print(f"Churn: {df['Churn'].eq('Yes').mean():.2%}")

    print("\n=== Auditoria semântica ===")
    print(audit.to_string(index=False))
    print("\nResumo:", audit_summary)

    result = run_pipeline(DATA_PATH)

    print("\n=== Seleção — somente treino / 5-fold CV ===")
    print(result["selection"].round(4).to_string(index=False))

    print("\nModelo selecionado:", result["selected_model"])

    print("\n=== Holdout final — threshold 0.50 ===")
    for key, value in result["test_metrics"].items():
        print(f"{key}: {value:.4f}")

    print("\n=== Campanha por capacidade — OOF do treino ===")
    print(result["top_k"].round(4).to_string(index=False))

    if result["coefficients"] is not None:
        print("\n=== Coeficientes de maior magnitude ===")
        print(
            result["coefficients"]
            [["feature", "coefficient", "odds_ratio"]]
            .head(12)
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
