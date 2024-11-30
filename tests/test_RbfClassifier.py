#!/usr/bin/env python
# Created by "Thieu" at 00:18, 01/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from evorbf import RbfClassifier


# Sample test data
@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_initialization():
    # Test if the RbfClassifier initializes with default parameters
    model = RbfClassifier()
    assert model.size_hidden == 10, "Default size_hidden should be 10"
    assert model.center_finder == "kmeans", "Default center_finder should be 'kmeans'"
    assert model.sigmas == 2.0, "Default sigmas should be 2.0"
    assert model.reg_lambda == 0.1, "Default reg_lambda should be 0.1"
    assert model.seed is None, "Default seed should be None"


def test_score(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RbfClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    assert isinstance(score, float), "Score should be a float"
    assert 0 <= score <= 1, "Score should be between 0 and 1"


def test_scores(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RbfClassifier()
    model.fit(X_train, y_train)
    metrics = model.scores(X_test, y_test, list_metrics=("AS", "RS"))
    assert isinstance(metrics, dict), "Scores should return a dictionary"
    assert all(isinstance(value, (int, float)) for value in metrics.values()), "Metric values should be int or float"


def test_evaluate(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RbfClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results = model.evaluate(y_test, y_pred, list_metrics=("AS", "RS"))
    assert isinstance(results, dict), "Evaluate should return a dictionary"
    assert all(isinstance(value, (int, float)) for value in results.values()), "Metric values should be int or float"


def test_fit_predict(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = RbfClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test), "The number of predictions should match the number of test samples"
    assert np.all(
        np.isin(predictions, np.unique(y_train))), "Predictions should only contain labels from the training set"
