import pytest
from fastapi.testclient import TestClient

from app import create_app


@pytest.fixture(scope="module")
def client(artifacts):
    with TestClient(create_app(artifacts["model"], artifacts["vocab"])) as c:
        yield c


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["vocab_size"] > 2


def test_index_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200 and "Next Word Predictor" in res.text


def test_cors_allows_the_local_frontend(client):
    res = client.options("/api/predict", headers={
        "Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_static_assets_are_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_predict_returns_five_suggestions(client):
    res = client.post("/api/predict", json={"text": "i want to "})
    assert res.status_code == 200
    body = res.json()
    assert len(body["suggestions"]) == 5 and body["mode"] == "next" and body["latency_ms"] >= 0


def test_predict_detects_completion_mode(client):
    body = client.post("/api/predict", json={"text": "i want to g"}).json()
    assert body["mode"] == "complete"
    assert all(s["word"].startswith("g") for s in body["suggestions"])


def test_predict_respects_k(client):
    assert len(client.post("/api/predict", json={"text": "hello ", "k": 3}).json()["suggestions"]) == 3


@pytest.mark.parametrize("payload", [{"text": "hi", "k": 0}, {"text": "hi", "k": 99}, {"text": "x" * 501}, {"k": "a"}])
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_empty_text_is_allowed(client):
    assert client.post("/api/predict", json={"text": ""}).status_code == 200


def test_api_degrades_gracefully_without_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.keras", tmp_path / "missing.json")) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/api/predict", json={"text": "hi"}).status_code == 503
        assert c.get("/").status_code == 200  # UI still renders and shows a notice