#!/usr/bin/env python
# Created by "Thieu" at 00:24, 01/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
from sklearn.datasets import make_regression
from evorbf import NiaRbfRegressor


@pytest.fixture
def sample_data():
    # Generate sample regression data
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    return X, y


def test_initialization():
    # Test that the class initializes correctly with default parameters
    model = NiaRbfRegressor(size_hidden=10, center_finder="kmeans", regularization=True,
                            optim="BaseGA", optim_paras={"epoch": 10, "pop_size": 30}, verbose=True, seed=42)

    assert model.size_hidden == 10
    assert model.center_finder == "kmeans"
    assert model.regularization is True
    assert model.optimizer.name == "BaseGA"
    assert model.optim_paras == {"epoch": 10, "pop_size": 30}
    assert model.verbose is True
    assert model.seed == 42


def test_fit_predict(sample_data):
    X, y = sample_data
    model = NiaRbfRegressor(size_hidden=10, center_finder="kmeans", regularization=False, obj_name="MSE",
                            optim="BaseGA", optim_paras={"epoch": 10, "pop_size": 30}, verbose=True, seed=42)

    # Fit the model
    model.fit(X, y)

    # Test that the model can make predictions
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape  # Ensure the output shape matches the input y shape


def test_score(sample_data):
    X, y = sample_data
    model = NiaRbfRegressor(size_hidden=10, center_finder="kmeans", regularization=False, obj_name="MSE",
                            optim="BaseGA", optim_paras={"epoch": 10, "pop_size": 30}, verbose=True, seed=42)

    # Fit the model
    model.fit(X, y)

    # Calculate the score
    score = model.score(X, y)
    assert isinstance(score, float)  # Score should be a float
    assert score >= 0  # R^2 score should be non-negative for a good model


def test_scores(sample_data):
    X, y = sample_data
    model = NiaRbfRegressor(size_hidden=10, center_finder="kmeans", regularization=False, obj_name="MSE",
                            optim="BaseGA", optim_paras={"epoch": 10, "pop_size": 30}, verbose=True, seed=42)

    # Fit the model
    model.fit(X, y)

    # Get multiple metrics
    metrics = model.scores(X, y, list_metrics=["MSE", "MAE"])
    assert isinstance(metrics, dict)  # The output should be a dictionary
    assert "MSE" in metrics  # Ensure MSE is in the returned metrics
    assert "MAE" in metrics  # Ensure MAE is in the returned metrics


def test_evaluate(sample_data):
    X, y = sample_data
    model = NiaRbfRegressor(size_hidden=10, center_finder="kmeans", regularization=False, obj_name="MSE",
                            optim="BaseGA", optim_paras={"epoch": 10, "pop_size": 30}, verbose=True, seed=42)

    # Fit the model
    model.fit(X, y)

    # Evaluate predictions
    y_pred = model.predict(X)
    metrics = model.evaluate(y, y_pred, list_metrics=["MSE", "MAE"])

    assert isinstance(metrics, dict)  # The output should be a dictionary
    assert "MSE" in metrics  # Ensure MSE is in the returned metrics
    assert "MAE" in metrics  # Ensure MAE is in the returned metrics
