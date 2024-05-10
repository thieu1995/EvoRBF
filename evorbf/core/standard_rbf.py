#!/usr/bin/env python
# Created by "Thieu" at 18:33, 10/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from evorbf.core.base_rbf import BaseRbf, RBF
from evorbf.helpers.scaler import ObjectiveScaler


class RbfRegressor(BaseRbf, RegressorMixin):
    """
    Defines the RBF model for Regression problems that inherit the BaseRbf and RegressorMixin classes.

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid",
        "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink",
        "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" }, default='sigmoid'
        Activation function for the hidden layer.

    seed: int, default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.

    Examples
    --------
    >>> from evorbf import RbfRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = RbfRegressor(hidden_size=10, act_name="elu")
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    """

    def __init__(self, size_hidden=10, center_finder="kmean", sigmas=2.0, regularization=False, lamda=0.01, seed=None):
        super().__init__(size_hidden, center_finder, sigmas, regularization, lamda, seed)

    def create_network(self, X, y):
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                size_output = 1
            elif y.ndim == 2:
                size_output = y.shape[1]
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector or 2D matrix.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        obj_scaler = ObjectiveScaler(obj_name="self", ohe_scaler=None)
        network = RBF(size_hidden=self.size_hidden, center_finder=self.center_finder, sigmas=self.sigmas,
                      regularization=self.regularization, seed=self.seed)
        return network, obj_scaler

    def score(self, X, y, method="RMSE"):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseRbf__score_reg(X, y, method)

    def scores(self, X, y, list_methods=("MSE", "MAE")):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseRbf__scores_reg(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("MSE", "MAE")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseRbf__evaluate_reg(y_true, y_pred, list_metrics)


class RbfClassifier(BaseRbf, ClassifierMixin):
    """
    Defines the general class of Metaheuristic-based RBF model for Classification problems that inherit the BaseRbf and ClassifierMixin classes.

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid",
        "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink",
        "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" }, default='sigmoid'
        Activation function for the hidden layer.

    seed: int, default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.

    Examples
    --------
    >>> from evorbf import Data, RbfClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = RbfClassifier(hidden_size=10, act_name="elu")
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    array([1, 0, 1, 0, 1])
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, size_hidden=10, center_finder="kmean", sigmas=2.0, regularization=False, lamda=0.01, seed=None):
        super().__init__(size_hidden, center_finder, sigmas, regularization, lamda, seed)
        self.return_prob = False
        self.n_labels = None

    def create_network(self, X, y):
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                self.n_labels = len(np.unique(y))
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        ohe_scaler = OneHotEncoder(sparse_output=False)
        ohe_scaler.fit(np.reshape(y, (-1, 1)))
        obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        network = RBF(size_hidden=self.size_hidden, center_finder=self.center_finder, sigmas=self.sigmas,
                      regularization=self.regularization, seed=self.seed)
        return network, obj_scaler

    def score(self, X, y, method="AS"):
        """
        Return the metric on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        method : str, default="AS"
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseRbf__score_cls(X, y, method)

    def scores(self, X, y, list_methods=("AS", "RS")):
        """
        Return the list of metrics on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        list_methods : list, default=("AS", "RS")
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseRbf__scores_cls(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Return the list of performance metrics on the given test data and labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("AS", "RS")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseRbf__evaluate_cls(y_true, y_pred, list_metrics)
