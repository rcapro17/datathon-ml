from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CoerceNumericTransformer(BaseEstimator, TransformerMixin):
    """
    Converte colunas para numérico (float) com errors='coerce'.
    Mantém o formato DataFrame (importante para ColumnTransformer).
    """

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        out = X.copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
