import pandas as pd

from src.transformers import CoerceNumericTransformer


def test_coerce_numeric_transformer_converts_values_and_keeps_dataframe():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "x"],
            "b": ["3.5", None, "4.2"],
        }
    )

    t = CoerceNumericTransformer()
    out = t.fit_transform(df)

    assert isinstance(out, pd.DataFrame)

    # "x" vira NaN (NaN não é comparável com ==)
    assert out["a"].iloc[0] == 1.0
    assert out["a"].iloc[1] == 2.0
    assert pd.isna(out["a"].iloc[2])

    assert out["b"].iloc[0] == 3.5
    assert pd.isna(out["b"].iloc[1])
    assert out["b"].iloc[2] == 4.2
