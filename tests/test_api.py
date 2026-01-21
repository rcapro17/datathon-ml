import json
from app.main import create_app

def test_health_endpoint():
    app = create_app()
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"

def test_predict_missing_features():
    app = create_app()
    client = app.test_client()
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data

def test_predict_requires_json_body():
    app = create_app()
    client = app.test_client()
    resp = client.post("/predict", data="nao-json", content_type="text/plain")
    assert resp.status_code == 400
    assert "error" in resp.get_json()

def test_predict_rejects_non_json(client):
    r = client.post("/predict", data="nao é json", content_type="text/plain")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_handles_exception(client, monkeypatch):
    import app.routes as routes

    def boom(_payload):
        raise RuntimeError("falhou")

    monkeypatch.setattr(routes, "predict", boom)

    r = client.post("/predict", json={"features": {"x": 1}})
    assert r.status_code == 400
    assert "falhou" in r.get_json()["error"]

def test_predict_logs_request_and_returns_200(client, monkeypatch, tmp_path):
    import app.routes as routes

    # Redireciona logs para uma pasta temporária
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Ajuste esses nomes conforme existirem no seu app/routes.py:
    # - se você tiver algo como LOG_DIR / REQUESTS_LOG_PATH / REQUESTS_PARQUET, etc.
    if hasattr(routes, "LOG_DIR"):
        monkeypatch.setattr(routes, "LOG_DIR", logs_dir, raising=False)
    if hasattr(routes, "REQUESTS_LOG_PATH"):
        monkeypatch.setattr(routes, "REQUESTS_LOG_PATH", logs_dir / "requests.parquet", raising=False)

    payload = {"features": {"INSTITUICAO_ENSINO_ALUNO_2020": "Escola Pública"}}

    r = client.post("/predict", json=payload)

    assert r.status_code == 200
    data = r.get_json()
    assert "predictions" in data
    assert "n" in data and data["n"] == 1




