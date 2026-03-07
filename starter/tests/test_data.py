import pandas as pd

from starter.starter.ml.data import process_data


def test_process_data_training_shapes():
    data = pd.DataFrame(
        {
            "age": [25, 30, 45],
            "workclass": ["Private", "Self-emp-not-inc", "Private"],
            "education": ["Bachelors", "HS-grad", "Masters"],
            "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
            "occupation": ["Tech-support", "Sales", "Exec-managerial"],
            "relationship": ["Not-in-family", "Husband", "Unmarried"],
            "race": ["White", "White", "Black"],
            "sex": ["Male", "Female", "Female"],
            "native-country": ["United-States", "United-States", "Canada"],
            "salary": ["<=50K", ">50K", ">50K"],
        }
    )

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    assert X.shape[0] == 3
    assert y.shape[0] == 3
    assert encoder is not None
    assert lb is not None
