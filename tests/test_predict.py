import pandas as pd

from src.predict import (
    load_schema,
    make_dataframe,
    normalize_features,
    predict,
)


def test_schema_loads():
    schema = load_schema()
    assert "features" in schema
    assert isinstance(schema["features"], list)
    assert len(schema["features"]) > 0


def test_make_dataframe_adds_missing_cols():
    schema = load_schema()
    rows = [{"INSTITUICAO_ENSINO_ALUNO_2020": "Escola Pública"}]  # incompleto
    df = make_dataframe(rows, schema)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == len(schema["features"])
    assert "INSTITUICAO_ENSINO_ALUNO_2020" in df.columns


def test_normalize_features_accepts_dict_and_list():
    payload1 = {"features": {"a": 1}}
    out1 = normalize_features(payload1)
    assert isinstance(out1, list) and isinstance(out1[0], dict)

    payload2 = {"features": [{"a": 1}, {"a": 2}]}
    out2 = normalize_features(payload2)
    assert len(out2) == 2


def test_normalize_features_rejects_missing_features_key():
    try:
        normalize_features({"x": 1})
        assert False, "Era para lançar erro"
    except ValueError as e:
        assert "features" in str(e).lower()


def test_normalize_features_rejects_wrong_features_type():
    try:
        normalize_features({"features": "invalido"})  # type: ignore
        assert False, "Era para lançar erro"
    except ValueError as e:
        assert "features" in str(e).lower()


def test_make_dataframe_drops_extra_columns():
    schema = load_schema()
    rows = [{"INSTITUICAO_ENSINO_ALUNO_2020": "Escola Pública", "COLUNA_EXTRA": 123}]
    df = make_dataframe(rows, schema)

    assert "COLUNA_EXTRA" not in df.columns
    assert df.shape[1] == len(schema["features"])


def test_predict_returns_expected_keys():
    payload = {"features": {"INSTITUICAO_ENSINO_ALUNO_2020": "Escola Pública"}}
    out = predict(payload)

    assert out["n"] == 1
    assert isinstance(out["predictions"], list) and len(out["predictions"]) == 1
    assert "probabilities" in out
    assert "model_info" in out
