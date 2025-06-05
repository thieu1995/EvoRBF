#!/usr/bin/env python
# Created by "Thieu" at 18:33, 10/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from evorbf.core.base_rbf import BaseRbf, CustomRBF
from evorbf.helpers.scaler import ObjectiveScaler, OneHotEncoder


class RbfRegressor(BaseRbf, RegressorMixin):
    """
    Implements a Radial Basis Function (RBF) model for regression tasks.

    This class extends `BaseRbf` and `RegressorMixin` to provide a regression model that:
        - Uses a non-linear Gaussian kernel function.
        - Computes output weights using inverse matrix multiplication.
        - Incorporates L2 regularization with the `reg_lambda` parameter.

    Parameters
    ----------
    size_hidden : int, default=10
        Number of hidden nodes in the RBF network.

    center_finder : str, default="kmeans"
        Method for finding the cluster centers. Options include methods like "random" or "kmeans".

    sigmas : float, default=2.0
        The width (sigma) of the Gaussian kernel.
        - In the standard RBF model, a single sigma value is used for all hidden nodes.
        - In nature-inspired algorithms (NIAs)-based RBF models, each hidden node can have a unique sigma value.

    reg_lambda : float, default=0.1
        Regularization parameter for L2 regularization. Set to `0` to disable regularization.

    seed : int, default=None
        Random seed for reproducibility in center initialization and weight generation.

    Examples
    --------
    >>> from evorbf import RbfRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = RbfRegressor(size_hidden=10, center_finder="kmeans", sigmas=2.0, reg_lambda=0.1, seed=42)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)

    Methods
    -------
    create_network(X, y):
        Constructs the RBF network based on the input data and labels.

    score(X, y):
        Calculates the R^2 (R-squared) score for the model's predictions.

    scores(X, y, list_metrics=("MSE", "MAE")):
        Computes a list of specified performance metrics (e.g., MSE, MAE) for the model's predictions.

    evaluate(y_true, y_pred, list_metrics=("MSE", "MAE")):
        Evaluates the model's performance using true and predicted values, based on a specified list of metrics.
    """

    def __init__(self, size_hidden=10, center_finder="kmeans", sigmas=2.0, reg_lambda=0.1, seed=None):
        super().__init__(size_hidden, center_finder, sigmas, reg_lambda, seed)

    def create_network(self, X, y):
        if isinstance(y, (list, tuple, np.ndarray)):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                self.size_output = 1
            elif y.ndim == 2:
                self.size_output = y.shape[1]
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector or 2D matrix.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        obj_scaler = ObjectiveScaler(obj_name="self", ohe_scaler=None)
        network = CustomRBF(size_hidden=self.size_hidden, center_finder=self.center_finder,
                            sigmas=self.sigmas, reg_lambda=self.reg_lambda, seed=self.seed)
        return network, obj_scaler

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


class RbfClassifier(BaseRbf, ClassifierMixin):
    """
    A Radial Basis Function (RBF) model for classification tasks.

    This class implements an RBF classifier that uses non-linear Gaussian activation functions
    and inverse matrix multiplication for output weight calculation. It allows for regularization
    through an L2 term.

    Parameters
    ----------
    size_hidden : int, default=10
        The number of hidden nodes in the RBF network.

    center_finder : str, default="kmeans"
        The method used to find cluster centers. For example, "kmeans" applies the K-Means clustering algorithm.

    sigmas : float, default=2.0
        The sigma value for the Gaussian activation function. In a traditional RBF model, one sigma is used for all
        hidden nodes. In Nature-inspired Algorithm-based RBF models, each hidden node may have its own sigma value.

    reg_lambda : float, default=0.1
        The regularization parameter for the L2 regularization term. Setting this to 0 disables regularization.

    seed : int or None, default=None
        The random seed for initializing the weights and biases. Specify an integer value for reproducibility.

    Attributes
    ----------
    n_labels : int
        The number of unique labels in the target dataset. Determined during the creation of the network.

    Examples
    --------
    >>> from evorbf import Data, RbfClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> model = RbfClassifier(size_hidden=10, center_finder="kmeans", sigmas=2.0, reg_lambda=0.1, seed=None)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    array([1, 0, 1, 0, 1])

    Notes
    -----
    - This class requires the Permetrics library to evaluate performance metrics.
    - One-hot encoding is applied internally to handle multi-class classification problems.
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, size_hidden=10, center_finder="kmeans", sigmas=2.0, reg_lambda=0.1, seed=None):
        super().__init__(size_hidden, center_finder, sigmas, reg_lambda, seed)
        self.n_labels, self.classes_ = None, None

    def create_network(self, X, y):
        """
        Initializes the RBF network and prepares data for classification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target labels. Must be a 1D array or vector containing integer class labels.

        Returns
        -------
        network : CustomRBF
            The initialized RBF network with the specified configuration.
        obj_scaler : ObjectiveScaler
            The scaler that applies one-hot encoding and the softmax activation function.

        Raises
        ------
        TypeError
            If the target array `y` is not a 1D array or is of an unsupported type.
        """
        if isinstance(y, (list, tuple, np.ndarray)):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                self.n_labels, self.classes_ = len(np.unique(y)), np.unique(y)
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        ohe_scaler = OneHotEncoder()
        ohe_scaler.fit(np.reshape(y, (-1, 1)))
        obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        network = CustomRBF(size_hidden=self.size_hidden, center_finder=self.center_finder,
                            sigmas=self.sigmas, reg_lambda=self.reg_lambda, seed=self.seed)
        return network, obj_scaler

    def score(self, X, y):
        """
        Calculates the accuracy of the model on the given test data and labels.

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
            The accuracy score of the classifier.
        """
        return self._BaseRbf__score_cls(X, y)

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """
        Computes multiple performance metrics for the given test data and labels.

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
             A dictionary where keys are metric names and values are the computed metric scores.
        """
        return self._BaseRbf__scores_cls(X, y, list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Evaluates performance metrics for predicted values compared to true labels.

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
            A dictionary where keys are metric names and values are the computed metric scores.
        """
        return self._BaseRbf__evaluate_cls(y_true, y_pred, list_metrics)
