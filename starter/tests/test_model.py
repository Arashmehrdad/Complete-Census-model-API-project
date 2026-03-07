import numpy as np

from starter.starter.ml.model import compute_model_metrics, inference, train_model


def test_train_model_returns_model():
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")


def test_compute_model_metrics_returns_floats():
    y = np.array([1, 0, 1, 1])
    preds = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference_returns_correct_length():
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert len(preds) == len(X_train)
