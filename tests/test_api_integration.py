import pytest

from app.predict import THRESHOLD


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_predict_ok(client):
    r = client.post(
        "/predict",
        json={
            "text": "You are great person with charming personality and good attitude."
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"label", "prob", "low_confidence"}
    assert 0.0 <= body["prob"] <= 1.0


def test_predict_short_text_low_confidence(client):
    r = client.post("/predict", json={"text": "ok"})
    assert r.status_code == 200
    assert r.json()["low_confidence"] is True


def test_predict_toxic_example(client):
    r = client.post(
        "/predict", json={"text": "You are the worst person i ever met, you idiot!"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("toxic", "clean")


def test_predict_invalid_payload(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422


@pytest.mark.parametrize("bad", [123, None, [], {}])
def test_predict_wrong_type(client, bad):
    r = client.post("/predict", json={"text": bad})
    assert r.status_code == 422


def test_feedback_ok(client):
    r = client.post(
        "/feedback",
        json={
            "text": "You are great person with charming personality and good attitude.",
            "true_label": 0,
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "stored"
