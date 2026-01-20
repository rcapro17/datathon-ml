import json
from pathlib import Path

import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"
RAW_PATH = Path("data/raw/PEDE_PASSOS_DATASET_FIAP.csv")
SCHEMA_PATH = Path("artifacts/schema.json")


def to_json_safe_dict(row: pd.Series) -> dict:
    """
    Converte NaN/NaT para None para ficar JSON-compliant.
    """
    d = row.to_dict()
    for k, v in d.items():
        if pd.isna(v):
            d[k] = None
    return d


def main():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = schema["features"]

    df = pd.read_csv(RAW_PATH, sep=";")[features].dropna(how="all")

    n = min(20, len(df))
    print(f"[INFO] Enviando {n} requests para gerar logs...")

    ok = 0
    for i in range(n):
        row = to_json_safe_dict(df.iloc[i])
        payload = {"features": row}

        r = requests.post(API_URL, json=payload, timeout=15)
        if r.status_code == 200:
            ok += 1
        else:
            print(f"[WARN] request {i} falhou: {r.status_code} {r.text}")

    print(f"[OK] Requests enviadas com sucesso: {ok}/{n}")
    print("[INFO] Verifique monitoring/logs/requests.parquet")


if __name__ == "__main__":
    main()
