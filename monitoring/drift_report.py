from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftTable


RAW_PATH = Path("data/raw/PEDE_PASSOS_DATASET_FIAP.csv")
SCHEMA_PATH = Path("artifacts/schema.json")
REQUEST_LOG = Path("monitoring/logs/requests.parquet")
REPORT_OUT = Path("monitoring/reports/drift_report.html")


def load_schema_features() -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["features"]


def load_reference_df(features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, sep=";")
    df = df[features].dropna(how="all")
    return df


def load_current_df(features: list[str]) -> pd.DataFrame:
    if not REQUEST_LOG.exists():
        raise FileNotFoundError(
            "Ainda não existem requests logadas em monitoring/logs/requests.parquet. "
            "Faça algumas chamadas no /predict primeiro."
        )

    df = pd.read_parquet(REQUEST_LOG)

    # garante que todas colunas do schema existam
    for col in features:
        if col not in df.columns:
            df[col] = None

    df = df[features].dropna(how="all")
    return df


def main():
    features = load_schema_features()
    ref = load_reference_df(features)
    cur = load_current_df(features)

    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=ref, current_data=cur)

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_OUT))

    print(f"[OK] Drift report gerado em: {REPORT_OUT}")


if __name__ == "__main__":
    main()
