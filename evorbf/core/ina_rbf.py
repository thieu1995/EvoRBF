#!/usr/bin/env python
# Created by "Thieu" at 17:03, 22/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
from typing import Tuple

import numpy as np
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder

from evorbf.core.base_rbf import BaseInaRbf, RBF
from evorbf.helpers.scaler import ObjectiveScaler


class InaRbfRegressor(BaseInaRbf, RegressorMixin):
    """
    Defines the general class of Metaheuristic-based RBF core for Regression problems that inherit the BaseInaRbf and RegressorMixin classes.

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink" }, default='sigmoid'
        Activation function for the hidden layer.

    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of
        optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Examples
    --------
    >>> from evorbf import InaRbfRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    >>> model = InaRbfRegressor(size_hidden=10, center_finder="kmean", sigmas=2.0, regularization=False, lamda=0.01,
    >>>         obj_name=None, optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True, seed=42, obj_weights=None)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    """

    def __init__(self, size_hidden=10, center_finder="kmean", regularization=False, lamda=0.01,
                 obj_name=None, optimizer="BaseGA", optimizer_paras=None, verbose=True, seed=None, obj_weights=None):
        super().__init__(size_hidden=size_hidden, center_finder=center_finder, regularization=regularization,
                         lamda=lamda, obj_name=obj_name, optimizer=optimizer, optimizer_paras=optimizer_paras, verbose=verbose, seed=seed)
        self.obj_weights = obj_weights

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
        if size_output > 1:
            if self.obj_weights is None:
                self.obj_weights = 1./size_output * np.ones(size_output)
            elif self.obj_weights in (list, tuple, np.ndarray):
                if not (len(self.obj_weights) == size_output):
                    raise ValueError(f"There is {size_output} objectives, but obj_weights has size of {len(self.obj_weights)}")
            else:
                raise TypeError("Invalid obj_weights array type, it should be list, tuple or np.ndarray")
        obj_scaler = ObjectiveScaler(obj_name="self", ohe_scaler=None)
        network = self._net_class(size_hidden=self.size_hidden, center_finder=self.center_finder, lamda=self.lamda)
        return network, obj_scaler

    def objective_function(self, solution=None):
        """
        Evaluates the fitness function for regression metric

        Parameters
        ----------
        solution : np.ndarray, default=None

        Returns
        -------
        result: float
            The fitness value
        """
        self.network.update_weights_from_solution(solution, self.X_temp, self.y_temp)
        y_pred = self.network.predict(self.X_temp)
        loss_train = RegressionMetric(self.y_temp, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train

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


class InaRbfClassifier(BaseInaRbf, ClassifierMixin):
    """
    Defines the general class of Metaheuristic-based RBF model for Classification problems that inherit the BaseInaRbf and ClassifierMixin classes.

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink" }, default='sigmoid'
        Activation function for the hidden layer.

    Examples
    --------
    >>> from evorbf import Data, InaRbfClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = InaRbfClassifier(hidden_size=10, act_name="elu")
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    array([1, 0, 1, 0, 1])
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, size_hidden=10, center_finder="kmean", regularization=False, lamda=0.01,
                 obj_name=None, optimizer="BaseGA", optimizer_paras=None, verbose=True, seed=None, obj_weights=None):
        super().__init__(size_hidden=size_hidden, center_finder=center_finder, regularization=regularization,
                         lamda=lamda, obj_name=obj_name, optimizer=optimizer, optimizer_paras=optimizer_paras, verbose=verbose, seed=seed)
        self.return_prob = False
        self.n_labels = None
        self.obj_weights = obj_weights

    def _check_y(self, y):
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                return len(np.unique(y))
            raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")

    def create_network(self, X, y) -> Tuple[RBF, ObjectiveScaler]:
        self.n_labels = self._check_y(y)
        if self.n_labels > 2:
            if self.obj_name in self.CLS_OBJ_LOSSES:
                self.return_prob = True
        ohe_scaler = OneHotEncoder(sparse_output=False)
        ohe_scaler.fit(np.reshape(y, (-1, 1)))
        obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        network = self._net_class(size_hidden=self.size_hidden, center_finder=self.center_finder,
                                  regularization=self.regularization, lamda=self.lamda, seed=self.seed)
        return network, obj_scaler

    def objective_function(self, solution=None):
        """
        Evaluates the fitness function for classification metric

        Parameters
        ----------
        solution : np.ndarray, default=None

        Returns
        -------
        result: float
            The fitness value
        """
        self.network.update_weights_from_solution(solution, self.X_temp, self.y_temp)
        y_pred = self.predict(self.X_temp, return_prob=self.return_prob)
        y1 = self.obj_scaler.inverse_transform(self.y_temp)
        loss_train = ClassificationMetric(y1, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train

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
