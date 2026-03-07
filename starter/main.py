import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


with open("starter/model/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("starter/model/encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

with open("starter/model/lb.pkl", "rb") as lb_file:
    lb = pickle.load(lb_file)


app = FastAPI()


class CensusData(BaseModel):
    age: int = Field(...)
    workclass: str = Field(...)
    fnlwgt: int = Field(...)
    education: str = Field(...)
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(...)
    relationship: str = Field(...)
    race: str = Field(...)
    sex: str = Field(...)
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlwgt": 34146,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )


@app.get("/")
def root():
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
def predict(data: CensusData):
    input_dict = {
        "age": data.age,
        "workclass": data.workclass,
        "fnlwgt": data.fnlwgt,
        "education": data.education,
        "education-num": data.education_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country,
    }

    df = pd.DataFrame([input_dict])

    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)[0]
    label = lb.classes_[pred]

    return {"prediction": label}

    return {"prediction": label}
