import pytest
from src.data_loader import load_and_split_data
from src.train_pipeline import train_and_evaluate

def test_data_loader_shapes():
    X_train, X_test, y_train, y_test = load_and_split_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_training_pipeline_runs():
    pipeline, mse, r2 = train_and_evaluate()
    assert mse > 0
    assert -1.0 <= r2 <= 1.0
    assert hasattr(pipeline, "predict")

def test_pipeline_prediction_output():
    pipeline, _, _ = train_and_evaluate()
    X_train, _, _, _ = load_and_split_data()
    preds = pipeline.predict(X_train[:5])
    assert len(preds) == 5
    assert all(isinstance(p, float) for p in preds)
