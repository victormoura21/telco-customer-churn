from pathlib import Path
import ast

root = Path(__file__).resolve().parent
required = [
    "README.md", "requirements.txt", "main.py",
    "src/data_preprocessing.py", "src/modeling.py",
    "notebooks/01_EDA.ipynb", "notebooks/02_Modeling.ipynb",
    "notebooks/03_Business_and_Interpretation.ipynb",
    "data/raw/telco_customer_churn.xlsx",
]
missing = [x for x in required if not (root/x).exists()]
if missing:
    raise SystemExit(f"Arquivos ausentes: {missing}")

for py in list((root/"src").glob("*.py")) + [root/"main.py"]:
    ast.parse(py.read_text(encoding="utf-8"))

print("Estrutura e sintaxe Python: OK")
