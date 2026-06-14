"""API smoke tests via FastAPI TestClient."""
from fastapi.testclient import TestClient
from api.main import app
from ml.data import make_synthetic


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_model_info():
    with TestClient(app) as client:
        r = client.get("/model/info")
        assert r.status_code == 200
        assert len(r.json()["classes"]) == 5


def test_predict():
    row = make_synthetic(1).drop(columns=["attack_class"]).iloc[0]
    features = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    with TestClient(app) as client:
        r = client.post("/predict", json={"features": features})
        assert r.status_code == 200
        body = r.json()
        assert body["attack_class"] in ["Normal", "DoS", "Probe", "R2L", "U2R"]
        assert 0.0 <= body["confidence"] <= 1.0