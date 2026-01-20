from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

MODEL_PATH = Path("app/model/model.joblib")
SCHEMA_PATH = Path("artifacts/schema.json")


def load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema não encontrado em {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def normalize_features(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Aceita:
      - {"features": {...}}            (um aluno)
      - {"features": [{...}, {...}]}   (lista de alunos)
    """
    if not isinstance(payload, dict):
        raise ValueError("JSON inválido: payload deve ser um objeto.")

    if "features" not in payload:
        raise ValueError("Campo obrigatório ausente: 'features'.")

    feats = payload["features"]
    if isinstance(feats, dict):
        return [feats]
    if isinstance(feats, list) and all(isinstance(x, dict) for x in feats):
        return feats

    raise ValueError("'features' deve ser um objeto ou uma lista de objetos.")


def make_dataframe(rows: List[Dict[str, Any]], schema: Dict[str, Any]) -> pd.DataFrame:
    expected_cols = schema["features"]
    df = pd.DataFrame(rows)

    # garante todas as colunas esperadas
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # remove extras e mantém ordem
    df = df[expected_cols]
    return df


def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    schema = load_schema()
    model = load_model()

    rows = normalize_features(payload)
    X = make_dataframe(rows, schema)

    preds = model.predict(X).tolist()
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1].tolist()

    return {
        "n": len(rows),
        "predictions": preds,
        "probabilities": probs,
        "model_info": {
            "target_binary": schema.get("target_binary"),
            "risk_rule": schema.get("risk_rule"),
        },
    }
