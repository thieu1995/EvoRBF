#!/usr/bin/env python
# Created by "Thieu" at 17:03, 22/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import Tuple
import numpy as np
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import RegressorMixin, ClassifierMixin
from evorbf.core.base_rbf import BaseNiaRbf, CustomRBF
from evorbf.helpers.scaler import ObjectiveScaler, OneHotEncoder


class NiaRbfRegressor(BaseNiaRbf, RegressorMixin):
    """
    Defines the general class of Nature-inspired Algorithm-based RBF models for Regression problems.
    It inherits the BaseNiaRbf and RegressorMixin (from scikit-learn library) classes.

    This class defines the InaRbf regressor model that:
        + Use NIA algorithm to find `sigmas` value and `weights` of output layer.
        + use non-linear Gaussian function with `sigmas` as standard deviation
        + set up regulation term with hyperparameter `regularization`

    Inherits:
        + BaseNiaRbf : The base class for NIA-based RBF networks.
        + RegressorMixin : Scikit-learn mixin class for regression estimators.

    Parameters
    ----------
    size_hidden : int, default=10
        The number of hidden nodes
    center_finder : str, default="kmeans"
        The method is used to find the cluster centers
    regularization : bool, default=True
        Determine if L2 regularization technique is used or not.
        If set to True, then the regularization lambda is learned during the training.
    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.
    optim : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.
    optim_params : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.
    verbose : bool, default=True
        Whether to print progress messages to stdout.
    seed : int, default=None
        The seed value is used for reproducibility.
    lb : int, float, tuple, list, np.ndarray, optional
        Lower bounds for sigmas in network.
    ub : int, float, tuple, list, np.ndarray, optional
        Upper bounds for sigmas in network.
    mode : str, optional
        Mode for optimizer (default is 'single').
    n_workers : int, optional
        Number of workers for parallel processing in optimizer (default is None).
    termination : any, optional
        Termination criteria for optimizer (default is None).

    Attributes
    ----------
    network : object
        The RBF network instance created during training.
    obj_weights : array-like
        Weights assigned to output objectives for multi-objective tasks.
    X_temp : array-like
        Training input data temporarily stored during optimization.
    y_temp : array-like
        Training output data temporarily stored during optimization.

    Methods
    -------
    fit(X, y)
        Fit the NIA-RBF model to the training data.
    predict(X)
        Predict regression outputs for the given input data.
    score(X, y)
        Return the R2 score for the model predictions on the given data.
    scores(X, y, list_metrics=("MSE", "MAE"))
        Calculate and return a dictionary of performance metrics for the given data.
    evaluate(y_true, y_pred, list_metrics=("MSE", "MAE"))
        Evaluate and return a dictionary of performance metrics for the provided true
        and predicted labels.
    objective_function(solution=None)
        Evaluate the fitness function for regression metrics using the given solution.
    create_network(X, y)
        Initialize and configure the RBF network structure based on the input data.

    Examples
    --------
    >>> from evorbf import NiaRbfRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    >>> model = NiaRbfRegressor(size_hidden=10, center_finder="kmeans", regularization=False,
    >>>         obj_name=None, optim="BaseGA", optim_params=opt_paras, verbose=True, seed=42)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)

    Notes
    -----
    - This class requires the `Mealpy` library for the metaheuristic optimization algorithms.
      More details and supported optimizers can be found at: https://github.com/thieu1995/mealpy.
    - For metrics, the `Permetrics` library is recommended: https://github.com/thieu1995/permetrics.

    """

    def __init__(self, size_hidden=10, center_finder="kmeans", regularization=False,
                 obj_name=None, optim="BaseGA", optim_params=None, verbose=True, seed=None,
                 lb=None, ub=None, mode='single', n_workers=None, termination=None):
        super().__init__(size_hidden=size_hidden, center_finder=center_finder, regularization=regularization,
                         obj_name=obj_name, optim=optim, optim_params=optim_params, verbose=verbose, seed=seed,
                         lb=lb, ub=ub, mode=mode, n_workers=n_workers, termination=termination)

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
        if self.regularization:
            network = self._net_class(size_hidden=self.size_hidden, center_finder=self.center_finder, seed=self.seed)
        else:
            network = self._net_class(self.size_hidden, self.center_finder, reg_lambda=0, seed=self.seed)
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

    def score(self, X, y):
        """Return the R2S metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseRbf__score_reg(X, y)

    def scores(self, X, y, list_metrics=("MSE", "MAE")):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_metrics : list, default=("MSE", "MAE")
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseRbf__scores_reg(X, y, list_metrics)

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


class NiaRbfClassifier(BaseNiaRbf, ClassifierMixin):
    """
    Defines the general class of Nature-inspired Algorithm-based RBF models for Classification problems.
    It inherits the BaseNiaRbf and ClassifierMixin (from scikit-learn library) classes.

    This class defines the InaRbf classifier model that:
        + Use NIA algorithm to find `sigmas` value and `weights` of output layer.
        + use non-linear Gaussian function with `sigmas` as standard deviation
        + set up regulation term with hyperparameter `regularization`

    Parameters
    ----------
    size_hidden : int, default=10
        The number of hidden nodes
    center_finder : str, default="kmeans"
        The method is used to find the cluster centers
    regularization : bool, default=True
        Determine if L2 regularization technique is used or not.
        If set to True, then the regularization lambda is learned during the training.
    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.
    optim : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.
    optim_params : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.
    verbose : bool, default=True
        Whether to print progress messages to stdout.
    seed : int, default=None
        The seed value is used for reproducibility.
    lb : int, float, tuple, list, np.ndarray, optional
        Lower bounds for sigmas in network.
    ub : int, float, tuple, list, np.ndarray, optional
        Upper bounds for sigmas in network.
    mode : str, optional
        Mode for optimizer (default is 'single').
    n_workers : int, optional
        Number of workers for parallel processing in optimizer (default is None).
    termination : any, optional
        Termination criteria for optimizer (default is None).

    Examples
    --------
    >>> from evorbf import Data, NiaRbfClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
    >>> print(NiaRbfClassifier.SUPPORTED_CLS_OBJECTIVES)
    >>> model = NiaRbfClassifier(size_hidden=25, center_finder="kmeans", regularization=False, obj_name="AS",
    >>>             optim="OriginalWOA", optim_params=opt_paras, verbose=True, seed=42)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    array([1, 0, 1, 0, 1])
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, size_hidden=10, center_finder="kmeans", regularization=False,
                 obj_name=None, optim="BaseGA", optim_params=None, verbose=True, seed=None,
                 lb=None, ub=None, mode='single', n_workers=None, termination=None):
        super().__init__(size_hidden=size_hidden, center_finder=center_finder, regularization=regularization,
                         obj_name=obj_name, optim=optim, optim_params=optim_params, verbose=verbose, seed=seed,
                         lb=lb, ub=ub, mode=mode, n_workers=n_workers, termination=termination)
        self.return_prob = False
        self.n_labels, self.classes_ = None, None

    def _check_y(self, y):
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                return len(np.unique(y)), np.unique(y)
            raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")

    def create_network(self, X, y) -> Tuple[CustomRBF, ObjectiveScaler]:
        self.n_labels, self.classes_ = self._check_y(y)
        if self.n_labels > 2:
            if self.obj_name in self.CLS_OBJ_LOSSES:
                self.return_prob = True
        ohe_scaler = OneHotEncoder()
        ohe_scaler.fit(np.reshape(y, (-1, 1)))
        obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        if self.regularization:
            network = self._net_class(size_hidden=self.size_hidden, center_finder=self.center_finder, seed=self.seed)
        else:
            network = self._net_class(self.size_hidden, self.center_finder, reg_lambda=0, seed=self.seed)
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
        if self.return_prob:
            y_pred = self.predict_proba(self.X_temp)
        else:
            y_pred = self.predict(self.X_temp)
        y1 = self.obj_scaler.inverse_transform(self.y_temp)
        loss_train = ClassificationMetric(y1, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train

    def score(self, X, y):
        """
        Return the accuracy score on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseRbf__score_cls(X, y)

    def scores(self, X, y, list_metrics=("AS", "RS")):
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

        list_metrics : list, default=("AS", "RS")
            You can get all metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseRbf__scores_cls(X, y, list_metrics)

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
