#!/usr/bin/env python
# Created by "Thieu" at 00:37, 01/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from evorbf import AdvancedRbfRegressor

## Load data object
# total samples = 442, total features = 10
X, y = load_diabetes(return_X_y=True)

## Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# Fixture for creating an instance of AdvancedRbfRegressor
@pytest.fixture
def rbf_regressor():
    return AdvancedRbfRegressor(center_finder="random", finder_params={"n_centers": 10},
                                rbf_kernel="gaussian", kernel_params={"sigma": 1.5},
                                reg_lambda=0.1, has_bias=False, seed=42)


def test_fit(rbf_regressor):
    model = rbf_regressor.fit(X_train, y_train)
    assert model is not None, "The model should be returned after fitting."
    assert model.size_output == 1, "The output size should be 1 for 1D target data."


def test_score(rbf_regressor):
    rbf_regressor.fit(X_train, y_train)
    score = rbf_regressor.score(X_test, y_test)
    assert isinstance(score, float), "The score should be a float value."
    assert 0 <= score <= 1, "The R² score should be between 0 and 1."


def test_scores(rbf_regressor):
    rbf_regressor.fit(X_train, y_train)
    metrics = rbf_regressor.scores(X_test, y_test, list_metrics=("MSE", "MAE"))
    assert "MSE" in metrics, "MSE should be included in the metrics."
    assert "MAE" in metrics, "MAE should be included in the metrics."
    assert isinstance(metrics["MSE"], float), "MSE should be a float value."
    assert isinstance(metrics["MAE"], float), "MAE should be a float value."


def test_evaluate(rbf_regressor):
    rbf_regressor.fit(X_train, y_train)
    y_pred = rbf_regressor.predict(X_test)
    metrics = rbf_regressor.evaluate(y_test, y_pred, list_metrics=("MSE", "MAE"))
    assert "MSE" in metrics, "MSE should be included in the evaluation metrics."
    assert "MAE" in metrics, "MAE should be included in the evaluation metrics."
    assert isinstance(metrics["MSE"], float), "MSE should be a float value."
    assert isinstance(metrics["MAE"], float), "MAE should be a float value."


def test_predict(rbf_regressor):
    rbf_regressor.fit(X_train, y_train)
    y_pred = rbf_regressor.predict(X_test)
    assert np.all(np.isfinite(y_pred)), "Predictions should contain finite numbers."
