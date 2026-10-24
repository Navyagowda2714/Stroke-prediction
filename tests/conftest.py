import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# app.py loads model_pickle.pkl / scaler.pkl with relative paths
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def flask_app():
    os.environ.pop("GOOGLE_API_KEY", None)  # never call the LLM in tests
    import app as app_module

    app_module.app.config.update(TESTING=True)
    return app_module


@pytest.fixture()
def client(flask_app):
    return flask_app.app.test_client()


@pytest.fixture()
def high_risk_patient():
    return {
        "gender": "Male", "age": "80", "hypertension": "1", "disease": "1",
        "married": "Yes", "work": "Private", "residence": "Urban",
        "glucose": "230.5", "bmi": "34.0", "smoking": "smokes",
    }


@pytest.fixture()
def low_risk_patient():
    return {
        "gender": "Female", "age": "25", "hypertension": "0", "disease": "0",
        "married": "No", "work": "Private", "residence": "Rural",
        "glucose": "85.0", "bmi": "22.0", "smoking": "never smoked",
    }
