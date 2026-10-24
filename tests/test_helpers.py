import pytest


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.0, "Very Low Risk"),
        (0.09, "Very Low Risk"),
        (0.1, "Low Risk"),
        (0.3, "Moderate Risk"),
        (0.5, "High Risk"),
        (0.7, "Very High Risk"),
        (1.0, "Very High Risk"),
    ],
)
def test_risk_level_boundaries(flask_app, probability, expected):
    assert flask_app.get_risk_level_description(probability) == expected


def test_format_patient_data_is_human_readable(flask_app, high_risk_patient):
    text = flask_app.format_patient_data_for_llm(high_risk_patient)
    assert "Age: 80 years old" in text
    assert "Hypertension: Yes" in text
    assert "Heart Disease: Yes" in text
    assert "Smoking Status: Smokes" in text


def test_influential_factors_high_risk(flask_app, high_risk_patient):
    text = flask_app.get_influential_factors_for_llm(high_risk_patient, is_high_risk=True)
    for factor in ["older age", "hypertension", "heart disease", "glucose", "Body Mass Index", "smoking"]:
        assert factor in text


def test_influential_factors_low_risk(flask_app, low_risk_patient):
    text = flask_app.get_influential_factors_for_llm(low_risk_patient, is_high_risk=False)
    for factor in ["younger age", "absence of hypertension", "never having smoked"]:
        assert factor in text
