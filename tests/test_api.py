import pytest


def test_model_and_scaler_load(flask_app):
    assert flask_app.model is not None, "model_pickle.pkl failed to load"
    assert flask_app.scaler is not None, "scaler.pkl failed to load"
    assert flask_app.scaler.n_features_in_ == 19


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<form" in response.data or b"<html" in response.data.lower()


def test_predict_returns_probabilities(client, high_risk_patient):
    response = client.post("/predict", json=high_risk_patient)
    assert response.status_code == 200
    body = response.get_json()
    details = body["detailed_results"]
    assert details["no_stroke_probability"] + details["stroke_probability"] == pytest.approx(1.0)
    assert details["raw_prediction"] in (0, 1)
    assert body["is_high_risk"] == details["raw_prediction"]
    # Without GOOGLE_API_KEY the app must degrade gracefully instead of crashing
    assert "LLM functionality is disabled" in body["llm_explanation"]


def test_high_risk_patient_scores_higher_than_low_risk(client, high_risk_patient, low_risk_patient):
    high = client.post("/predict", json=high_risk_patient).get_json()
    low = client.post("/predict", json=low_risk_patient).get_json()
    assert (
        high["detailed_results"]["stroke_probability"]
        > low["detailed_results"]["stroke_probability"]
    )


def test_predict_rejects_empty_body(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_predict_handles_bad_input(client, low_risk_patient):
    bad = dict(low_risk_patient, age="not-a-number")
    response = client.post("/predict", json=bad)
    assert response.status_code >= 400
    assert "error" in response.get_json()
