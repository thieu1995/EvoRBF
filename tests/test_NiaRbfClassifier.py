#!/usr/bin/env python
# Created by "Thieu" at 00:29, 01/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import make_classification
from evorbf import NiaRbfClassifier, Data


@pytest.fixture
def sample_data():
    """Fixture to create sample data for testing."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    data = Data(X, y)
    data.split_train_test(test_size=0.2, random_state=42)
    return data


def test_classifier_initialization():
    """Test initialization of NiaRbfClassifier."""
    model = NiaRbfClassifier(size_hidden=20, center_finder="kmeans", regularization=True, seed=42,
                             optim="BaseGA", optim_params={"epoch": 30, "pop_size": 20})
    assert model.size_hidden == 20
    assert model.center_finder == "kmeans"
    assert model.regularization is True
    assert model.seed == 42


def test_classifier_fit_predict(sample_data):
    """Test fitting and predicting with NiaRbfClassifier."""
    model = NiaRbfClassifier(size_hidden=10, center_finder="kmeans", obj_name="F1S", regularization=True,
                             seed=42, optim="BaseGA", optim_params={"epoch": 30, "pop_size": 20})
    model.fit(sample_data.X_train, sample_data.y_train)
    predictions = model.predict(sample_data.X_test)
    assert predictions.shape == (sample_data.X_test.shape[0],)
    assert len(np.unique(predictions)) > 0  # Check there are predictions and not empty


def test_classifier_score(sample_data):
    """Test the score method of NiaRbfClassifier."""
    model = NiaRbfClassifier(size_hidden=10, center_finder="kmeans", regularization=True,
                             obj_name="F1S", seed=42,
                             optim="BaseGA", optim_params={"epoch": 30, "pop_size": 20})
    model.fit(sample_data.X_train, sample_data.y_train)
    score = model.score(sample_data.X_test, sample_data.y_test)
    assert isinstance(score, float)  # Score should be a float
    assert 0 <= score <= 1  # Score should be between 0 and 1 for accuracy


def test_classifier_metrics(sample_data):
    """Test the evaluation of metrics in NiaRbfClassifier."""
    model = NiaRbfClassifier(size_hidden=10, center_finder="kmeans", regularization=True,
                             obj_name="F1S", seed=42,
                             optim="BaseGA", optim_params={"epoch": 30, "pop_size": 20})
    model.fit(sample_data.X_train, sample_data.y_train)
    metrics = model.evaluate(sample_data.y_test, model.predict(sample_data.X_test), list_metrics=["AS", "RS"])
    assert isinstance(metrics, dict)  # Should return a dictionary of metrics
    assert "AS" in metrics  # Ensure that specific metrics are returned
    assert "RS" in metrics


def test_classifier_invalid_y():
    """Test the _check_y method with invalid input."""
    model = NiaRbfClassifier(size_hidden=10, center_finder="kmeans", regularization=True,
                             obj_name="F1S", seed=42,
                             optim="BaseGA", optim_params={"epoch": 30, "pop_size": 20})
    with pytest.raises(TypeError):
        model._check_y("invalid input")

    with pytest.raises(TypeError):
        model._check_y(np.array([[1, 2], [3, 4]]))  # Non-1D array


if __name__ == "__main__":
    pytest.main()
