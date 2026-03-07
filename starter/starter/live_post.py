import requests

url = "http://127.0.0.1:8000/predict"

payload = {
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
    "native-country": "United-States"
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response:", response.json())