from __future__ import annotations
from src.transformers import CoerceNumericTransformer



import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


DATASET_PATH = Path("data/raw/PEDE_PASSOS_DATASET_FIAP.csv")
MODEL_PATH = Path("app/model/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")
SCHEMA_PATH = Path("artifacts/schema.json")

TARGET_COL = "DEFASAGEM_2021"

# Baseline “justo”: usar apenas sinais até 2021 (evita olhar 2022)
ALLOWED_SUFFIXES = ("_2020",)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {DATASET_PATH}")

    try:
        df = pd.read_csv(DATASET_PATH, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATASET_PATH, sep=";", encoding="latin-1")

    return df


def make_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define risco de defasagem:
      - risk_defasagem=1 se DEFASAGEM_2021 < 0
      - risk_defasagem=0 se DEFASAGEM_2021 >= 0
    Remove linhas sem target (NaN).
    """
    df = df.copy()
    df = df[~df[TARGET_COL].isna()].copy()
    df["risk_defasagem"] = (df[TARGET_COL] < 0).astype(int)
    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Mantém apenas features permitidas (2020 e 2021) e remove colunas com risco de leakage.
    """
    cols = [c for c in df.columns if c.endswith(ALLOWED_SUFFIXES)]

    # remove target original
    cols = [c for c in cols if c != TARGET_COL]

    # remover NOME (alto risco de overfit / identificação)
    if "NOME" in cols:
        cols.remove("NOME")

    # remover colunas com alto risco de leakage / target leakage
    leakage_cols = {
        "NIVEL_IDEAL_2021",
        "FASE_2021",
    }
    cols = [c for c in cols if c not in leakage_cols]

    # regra extra: remove qualquer coluna que contenha termos suspeitos
    suspicious_terms = ("DEFASAGEM", "NIVEL_IDEAL")
    cols = [c for c in cols if not any(t in c.upper() for t in suspicious_terms)]

    X = df[cols].copy()
    return X, cols


def coerce_numeric_df(X: Any) -> pd.DataFrame:
    """
    Converte colunas para numérico (float) de forma robusta.
    - strings numéricas viram float
    - strings não numéricas viram NaN
    - mantém index
    """
    if isinstance(X, pd.DataFrame):
        df = X
    else:
        df = pd.DataFrame(X)

    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Pipeline robusto:
    - detecta colunas numéricas vs categóricas
    - numéricas: coerção -> imputação mediana
    - categóricas: imputação moda -> one-hot
    """
    # Descobrir quais colunas podem ser numéricas (tentando converter)
    X_try = X.copy()
    for c in X_try.columns:
        X_try[c] = pd.to_numeric(X_try[c], errors="coerce")

    numeric_features = [
        c
        for c in X_try.columns
        if pd.api.types.is_numeric_dtype(X_try[c]) and X_try[c].notna().sum() > 0
    ]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Importante: sem lambda (para joblib/pickle funcionar)
    to_numeric = FunctionTransformer(coerce_numeric_df, validate=False)
   


    numeric_transformer = Pipeline(
    steps=[
        ("to_numeric", CoerceNumericTransformer()),
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return clf


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    # ROC-AUC só faz sentido se tiver predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))

    return metrics


def save_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    df = load_dataset()
    df = make_binary_target(df)

    X, feature_cols = select_features(df)
    y = df["risk_defasagem"].values

    # split estratificado (porque pode ser desbalanceado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    # salvar modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # salvar schema (input esperado)
    schema = {
        "target_original": TARGET_COL,
        "target_binary": "risk_defasagem",
        "risk_rule": "risk=1 if DEFASAGEM_2021 < 0 else 0",
        "features": feature_cols,
        "n_rows_used": int(len(df)),
        "n_features": int(len(feature_cols)),
    }

    save_json(METRICS_PATH, metrics)
    save_json(SCHEMA_PATH, schema)

    print("\n[OK] Treino concluído!")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(f"Schema salvo em: {SCHEMA_PATH}")
    print("\nMétricas principais:")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if k in metrics:
            print(f" - {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
