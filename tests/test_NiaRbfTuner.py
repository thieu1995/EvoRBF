#!/usr/bin/env python
# Created by "Thieu" at 00:15, 01/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%


import pytest
from unittest.mock import MagicMock
# from nia_rbf_tuner import NiaRbfTuner, RbfRegressor, RbfClassifier, Optimizer, HyperparameterProblem
from evorbf import RbfRegressor, RbfClassifier, NiaRbfTuner
from mealpy import Optimizer
from permetrics import RegressionMetric, ClassificationMetric
from mealpy import WOA


# Mock dependencies to isolate the class functionality
@pytest.fixture
def mock_optimizer():
    return MagicMock(spec=Optimizer)


@pytest.fixture
def nia_rbf_tuner_regression():
    return NiaRbfTuner(
        problem_type="regression",
        bounds=[(0, 1), (1, 10)],
        cv=5,
        scoring="MSE",
        optim="OriginalWOA",
        optim_paras={"epoch": 100, "pop_size": 50},
        verbose=True,
        seed=42
    )


@pytest.fixture
def nia_rbf_tuner_classification():
    return NiaRbfTuner(
        problem_type="classification",
        bounds=[(0, 1), (1, 10)],
        cv=5,
        scoring="accuracy",
        optim="OriginalWOA",
        optim_paras={"epoch": 100, "pop_size": 50},
        verbose=True,
        seed=42
    )


def test_initialization_regression(nia_rbf_tuner_regression):
    assert nia_rbf_tuner_regression.problem_type == "regression"
    assert isinstance(nia_rbf_tuner_regression.network_class, RbfRegressor)
    assert nia_rbf_tuner_regression.scoring == "MSE"
    assert nia_rbf_tuner_regression.cv == 5
    assert nia_rbf_tuner_regression.verbose == "console"
    assert nia_rbf_tuner_regression.seed == 42


def test_initialization_classification(nia_rbf_tuner_classification):
    assert nia_rbf_tuner_classification.problem_type == "classification"
    assert isinstance(nia_rbf_tuner_classification.network_class, RbfClassifier)
    assert nia_rbf_tuner_classification.scoring == "accuracy"
    assert nia_rbf_tuner_classification.cv == 5
    assert nia_rbf_tuner_classification.verbose == "console"
    assert nia_rbf_tuner_classification.seed == 42


def test_set_optim(mock_optimizer):
    nia_rbf_tuner = NiaRbfTuner(
        problem_type="regression",
        bounds=[(0, 1)],
        cv=5,
        scoring="MSE",
        optim="OriginalWOA",
        optim_paras={"epoch": 100, "pop_size": 50},
        verbose=True,
        seed=42
    )

    # Mock the `get_optimizer_by_name` function to return a mock optimizer
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr('mealpy.optimizers.get_optimizer_by_name', lambda x: mock_optimizer)
        opt_instance = nia_rbf_tuner._set_optim("OriginalWOA", {"epoch": 100, "pop_size": 50})

    assert opt_instance == mock_optimizer
    mock_optimizer.set_parameters.assert_called_once_with({"epoch": 100, "pop_size": 50})


def test_fit_predict(nia_rbf_tuner_regression):
    # Mock training data
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [1, 2, 3]
    X_test = [[7, 8], [9, 10]]

    # Mock the training and prediction of the best estimator
    nia_rbf_tuner_regression.best_estimator = MagicMock()
    nia_rbf_tuner_regression.best_estimator.fit.return_value = None
    nia_rbf_tuner_regression.best_estimator.predict.return_value = [4, 5]

    # Mock the optimizer and problem to avoid actual computation
    nia_rbf_tuner_regression.optimizer = MagicMock()
    nia_rbf_tuner_regression.problem = MagicMock()

    nia_rbf_tuner_regression.fit(X_train, y_train)

    # Check that the best estimator was fitted
    nia_rbf_tuner_regression.best_estimator.fit.assert_called_once_with(X_train, y_train)

    # Check prediction method
    predictions = nia_rbf_tuner_regression.predict(X_test)
    assert predictions == [4, 5]


def test_predict_without_fit(nia_rbf_tuner_regression):
    with pytest.raises(ValueError, match="Model is not trained, please call the fit\\(\\) function."):
        nia_rbf_tuner_regression.predict([[1, 2], [3, 4]])


# Run tests with pytest
if __name__ == "__main__":
    pytest.main()


#
#
# import pytest
# import numpy as np
# from sklearn.datasets import make_regression
# from evorbf import NiaRbfTuner
#
#
# @pytest.fixture
# def data():
#     X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
#     return X, y
#
#
# def test_nia_rbf_tuner_initialization(data):
#     X, y = data
#     tuner = NiaRbfTuner(problem_type="regression", bounds=[(0.1, 10)], cv=5, scoring="MSE",
#                         optim="OriginalWOA", optim_paras={"epoch": 100, "pop_size": 20}, verbose=True, seed=42)
#     assert tuner.problem_type == "regression"
#     assert tuner.bounds == [(0.1, 10)]
#     assert tuner.cv == 5
#     assert tuner.scoring == "MSE"
#     assert tuner.seed == 42
#
#
# def test_nia_rbf_tuner_fit(data):
#     X, y = data
#     tuner = NiaRbfTuner(problem_type="regression", bounds=[(0.1, 10)], cv=5, scoring="MSE",
#                         optim="OriginalWOA", optim_paras={"epoch": 100, "pop_size": 20}, verbose=True, seed=42)
#     tuner.fit(X, y)
#     assert tuner.best_params is not None
#     assert tuner.best_estimator is not None
#     assert len(tuner.loss_train) > 0  # Ensure optimization history exists
#
#
# def test_nia_rbf_tuner_predict(data):
#     X, y = data
#     tuner = NiaRbfTuner(problem_type="regression", bounds=[(0.1, 10)], cv=5, scoring="MSE",
#                         optim="OriginalWOA", optim_paras={"epoch": 100, "pop_size": 20}, verbose=True, seed=42)
#     tuner.fit(X, y)
#     predictions = tuner.predict(X)
#     assert predictions.shape == (X.shape[0],)
#     assert np.isfinite(predictions).all()
