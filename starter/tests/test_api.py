from starter.main import app
from fastapi.testclient import TestClient
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


client = TestClient(app)

sample_low = {
    "age": 25,
    "workclass": "Private",
    "fnlwgt": 226802,
    "education": "11th",
    "education-num": 7,
    "marital-status": "Never-married",
    "occupation": "Machine-op-inspct",
    "relationship": "Own-child",
    "race": "Black",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

sample_high = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlwgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}


def test_root_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Prediction API"
    }


def test_predict_low_income():
    response = client.post("/predict", json=sample_low)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_predict_high_income():
    response = client.post("/predict", json=sample_high)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]
