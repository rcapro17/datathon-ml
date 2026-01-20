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

