import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import compute_model_metrics, inference, train_model


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


def compute_slice_metrics(
        test_data,
        cat_features,
        model,
        encoder,
        lb,
        output_path):
    """
    Compute model metrics for slices of categorical features and write them to a file.

    Inputs
    ------
    test_data : pd.DataFrame
        Test dataframe.
    cat_features : list[str]
        Categorical feature names.
    model :
        Trained model.
    encoder :
        Trained encoder.
    lb :
        Trained label binarizer.
    output_path : str
        File path for saving slice metrics.

    Returns
    -------
    None
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for feature in cat_features:
            for value in test_data[feature].unique():
                data_slice = test_data[test_data[feature] == value]

                X_slice, y_slice, _, _ = process_data(
                    data_slice,
                    categorical_features=cat_features,
                    label="salary",
                    training=False,
                    encoder=encoder,
                    lb=lb,
                )

                preds = inference(model, X_slice)
                precision, recall, fbeta = compute_model_metrics(
                    y_slice, preds)

                file.write(
                    f"Feature: {feature} | "
                    f"Value: {value} | "
                    f"Precision: {precision:.3f} | "
                    f"Recall: {recall:.3f} | "
                    f"F1: {fbeta:.3f}\n"
                )


if __name__ == "__main__":
    data = pd.read_csv("starter/data/census.csv", skipinitialspace=True)

    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {fbeta:.3f}")

    os.makedirs("starter/model", exist_ok=True)

    with open("starter/model/model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("starter/model/encoder.pkl", "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    with open("starter/model/lb.pkl", "wb") as lb_file:
        pickle.dump(lb, lb_file)

    compute_slice_metrics(
        test,
        CAT_FEATURES,
        model,
        encoder,
        lb,
        "starter/model/slice_output.txt",
    )
