#!/usr/bin/env python
# Created by "Thieu" at 00:10, 01/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import make_regression
from evorbf import RbfRegressor


@pytest.fixture
def data():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    return X, y


@pytest.fixture
def rbf_regressor():
    return RbfRegressor(size_hidden=10, center_finder="kmeans", sigmas=2.0, reg_lambda=0.1, seed=42)


def test_rbf_regressor_initialization(rbf_regressor):
    assert rbf_regressor.size_hidden == 10
    assert rbf_regressor.center_finder == "kmeans"
    assert rbf_regressor.sigmas == 2.0
    assert rbf_regressor.reg_lambda == 0.1
    assert rbf_regressor.seed == 42


def test_rbf_regressor_predict(rbf_regressor, data):
    X, y = data
    rbf_regressor.fit(X, y)
    predictions = rbf_regressor.predict(X)
    assert predictions.shape == (X.shape[0],)
    assert np.isfinite(predictions).all()


def test_rbf_regressor_score(rbf_regressor, data):
    X, y = data
    rbf_regressor.fit(X, y)
    score = rbf_regressor.score(X, y)
    assert isinstance(score, float)
    assert score >= 0.0  # Assuming R2 score is non-negative


def test_rbf_regressor_scores(rbf_regressor, data):
    X, y = data
    rbf_regressor.fit(X, y)
    metrics = rbf_regressor.scores(X, y, list_metrics=["MSE", "MAE"])
    assert isinstance(metrics, dict)
    assert "MSE" in metrics
    assert "MAE" in metrics


def test_rbf_regressor_evaluate(rbf_regressor, data):
    X, y = data
    rbf_regressor.fit(X, y)
    predictions = rbf_regressor.predict(X)
    metrics = rbf_regressor.evaluate(y, predictions, list_metrics=["MSE", "MAE"])
    assert isinstance(metrics, dict)
    assert "MSE" in metrics
    assert "MAE" in metrics
