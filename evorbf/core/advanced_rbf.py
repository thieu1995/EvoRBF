#!/usr/bin/env python
# Created by "Thieu" at 12:37, 29/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from permetrics import RegressionMetric, ClassificationMetric
from evorbf.helpers import kernel as km, center_finder as cfm
from evorbf.helpers.metrics import get_all_regression_metrics, get_all_classification_metrics
from evorbf.helpers.scaler import ObjectiveScaler, OneHotEncoder


class AdvancedRbfNet:
    """
    Implements an advanced Radial Basis Function (RBF) network for supervised learning tasks.

    This class provides a highly customizable RBF network with support for different kernel functions,
    center-finding strategies, regularization, and optional bias terms. It can be used for both regression
    and classification problems.

    Attributes:
        SUPPORT_KERNELS (dict): A dictionary of supported RBF kernel functions and their corresponding implementations.
            Options include:
                - "gaussian": Gaussian kernel
                - "multiquadric": Multiquadric kernel
                - "inverse_multiquadric": Inverse multiquadric kernel
                - "linear": Linear kernel
        SUPPORT_CENTER_FINDERS (dict): A dictionary of supported center-finding methods and their corresponding implementations.
            Options include:
                - "random": Random center finder
                - "kmeans": K-Means clustering
                - "meanshift": Mean-Shift clustering
                - "dbscan": DBSCAN clustering
        reg_lambda (float): L2 regularization strength (default is 0.1). If `reg_lambda=0`, no regularization is applied.
        has_bias (bool): Whether to include a bias term in the output layer (default is False).
        seed (int): Random seed for reproducibility.
        centers (np.ndarray): Array of RBF centers learned during training.
        sigmas (np.ndarray or None): Width parameters for the RBF functions (if applicable).
        weights (np.ndarray): Weights of the output layer learned during training.
        bias (float or None): Bias term learned during training (if `has_bias=True`).
        n_centers (int): Number of centers learned during training.

    Methods:
        __init__(center_finder, finder_params, rbf_kernel, kernel_params, reg_lambda, has_bias, seed):
            Initializes the RBF network with specified configurations for center finding, kernel functions,
            and regularization.
        fit(X, y):
            Fits the RBF network to the given training data and labels.
        predict(X):
            Predicts outputs for the given input data using the trained RBF network.

    Args:
        center_finder (str or CenterFinder): Method for finding RBF centers. Options include strings from `SUPPORT_CENTER_FINDERS`
            or an instance of a custom `CenterFinder` class. Default is "random".
        finder_params (dict, optional): Hyperparameters for the center-finding method. Defaults to `{"n_centers": 10}`.
        rbf_kernel (str or Kernel): RBF kernel function to use. Options include strings from `SUPPORT_KERNELS`
            or an instance of a custom `Kernel` class. Default is "gaussian".
        kernel_params (dict, optional): Hyperparameters for the RBF kernel function. Default is `None`.
        reg_lambda (float): L2 regularization strength. Default is 0.1.
        has_bias (bool): Whether to include a bias term in the output layer. Default is False.
        seed (int): Random seed for reproducibility. Default is 42.

    Raises:
        ValueError: If an unsupported center-finding or kernel method is provided.
        ValueError: If `reg_lambda` is not a float or `None`.

    Example:
        >>> import numpy as np
        >>> from evorbf import AdvancedRbfNet
        >>> X = np.random.rand(100, 2)
        >>> y = np.sin(X[:, 0]) + np.cos(X[:, 1])
        >>> rbf_net = AdvancedRbfNet(center_finder="kmeans", rbf_kernel="gaussian", reg_lambda=0.01, has_bias=True)
        >>> rbf_net.fit(X, y)
        >>> predictions = rbf_net.predict(X)
    """

    SUPPORT_KERNELS = {
        "gaussian": "GaussianKernel",
        "multiquadric": "MultiquadricKernel",
        "inverse_multiquadric": "InverseMultiquadricKernel",
        "linear": "LinearKernel",
    }

    SUPPORT_CENTER_FINDERS = {
        "random": "RandomFinder",
        "kmeans": "KMeansFinder",
        "meanshift": "MeanShiftFinder",
        "dbscan": "DbscanFinder",
    }

    def __init__(self, center_finder="random", finder_params=None,
                 rbf_kernel="gaussian", kernel_params=None, reg_lambda=0.1, has_bias=False, seed=42):
        """
        Args:
            center_finder: A string ("random" or "kmeans") or an instance of CenterFinder.
            finder_params: A dictionary of hyperparameters for center finder method. (Default: n_centers=10)
            rbf_kernel: A string ("gaussian", "multiquadric", etc.) or an instance of Kernel.
            kernel_params: Dictionary of hyperparameters for the RBF kernel.
            reg_lambda: L2 regularization strength (lambda). reg_lambda=0 then no regularization is applied.
            has_bias: Set up the bias in output layer or not.
            seed: The random seed value for reproducibility.
        """
        self.reg_lambda = reg_lambda
        self.has_bias = has_bias
        self.seed = seed

        # Configure the center-finding strategy
        if isinstance(center_finder, str):
            if center_finder in self.SUPPORT_CENTER_FINDERS.keys():
                self.center_finder = getattr(cfm, self.SUPPORT_CENTER_FINDERS[center_finder])
            else:
                raise ValueError(f"Unsupported center_finder method: {center_finder}")
        elif isinstance(center_finder, cfm.CenterFinder):
            self.center_finder = center_finder
        else:
            raise ValueError("center_finder must be a string or an instance of CenterFinder")
        self.center_finder = self.center_finder(**(finder_params or {}))

        # Configure the RBF function
        if isinstance(rbf_kernel, str):
            if rbf_kernel in self.SUPPORT_KERNELS.keys():
                self.rbf_kernel = getattr(km, self.SUPPORT_KERNELS[rbf_kernel])
            else:
                raise ValueError(f"Unsupported rbf_kernel method: {rbf_kernel}")
        elif isinstance(rbf_kernel, km.Kernel):
            self.rbf_kernel = rbf_kernel
        else:
            raise ValueError("rbf_kernel must be a string ('gaussian', 'multiquadric', etc.) or an instance of Kernel")
        self.rbf_kernel = self.rbf_kernel(**(kernel_params or {}))

        self.centers = None
        self.sigmas = None
        self.weights = None
        self.bias = None
        self.n_centers = None

    def fit(self, X, y):
        # Use the center finder to determine centers
        self.centers = self.center_finder.find_centers(X)
        self.n_centers = len(self.centers)

        # Construct the RBF matrix
        G = np.zeros((X.shape[0], self.n_centers))
        for i, c in enumerate(self.centers):
            G[:, i] = self.rbf_kernel.compute(X, c)

        if self.has_bias:
            # Add bias term to the RBF matrix
            G = np.hstack((G, np.ones((G.shape[0], 1))))  # Bias column

        # Solve for weights using regularized least squares
        if self.reg_lambda is None or self.reg_lambda == 0:
            params = np.linalg.pinv(G) @ y  # Moore-Penrose inverse for non-regularized case
        elif type(self.reg_lambda) is float:
            lambda_iden = self.reg_lambda * np.eye(G.shape[1])     # L2 regularization
            params = np.linalg.inv(G.T @ G + lambda_iden) @ G.T @ y
        else:
            raise ValueError("reg_lambda must be a float or None")

        # Split parameters into weights and bias
        if self.has_bias:
            self.weights = params[:-1]
            self.bias = params[-1]
        else:
            self.weights = params

    def predict(self, X):
        # Construct the RBF matrix for the test data
        G = np.zeros((X.shape[0], self.n_centers))
        for i, c in enumerate(self.centers):
            G[:, i] = self.rbf_kernel.compute(X, c)

        if self.has_bias:
            # Add bias term to the RBF matrix
            G = np.hstack((G, np.ones((G.shape[0], 1))))
            # Predict using the learned weights
            return G @ np.concatenate([self.weights, [self.bias]])
        else:
            return G @ self.weights


class BaseAdvancedRbf(BaseEstimator):
    """
    This class serves as a wrapper for an advanced Radial Basis Function (RBF) network, providing utility
    methods for training, evaluating, and saving the model and its associated data. It extends the `BaseEstimator` class.

    Attributes:
        SUPPORTED_CLS_METRICS (dict): Supported classification metrics, obtained from `get_all_classification_metrics`.
        SUPPORTED_REG_METRICS (dict): Supported regression metrics, obtained from `get_all_regression_metrics`.
        CLS_OBJ_LOSSES (list): Classification objective losses (defined by the user or inherited from subclasses).

    Methods:
        __init__: Initializes the RBF network with specified configurations.
        fit: Fits the RBF network to the given training data.
        predict: Predicts outputs for the provided input data using the trained RBF network.
        predict_proba: Predicts probabilities for classification tasks (if applicable).
        evaluate: Evaluates the model's predictions against true values using specified metrics.
        score: Computes a primary score (e.g., accuracy for classification or R² for regression) for the provided data.
        scores: Computes multiple metrics for the provided data.
        save_loss_train: Saves the training loss history to a CSV file.
        save_metrics: Saves computed evaluation metrics to a CSV file.
        save_y_predicted: Saves true and predicted values to a CSV file.
        save_model: Saves the model to a pickle file.
        load_model: Loads a model from a pickle file.

    Private Methods:
        __evaluate_reg: Evaluates regression metrics for given true and predicted values.
        __evaluate_cls: Evaluates classification metrics for given true and predicted values.
        __score_reg: Computes a regression score (e.g., R²) for the given data.
        __scores_reg: Computes multiple regression metrics for the given data.
        __score_cls: Computes a classification score (e.g., accuracy) for the given data.
        __scores_cls: Computes multiple classification metrics for the given data.

    Parameters for Initialization:
        center_finder (str): Method for finding RBF centers (default: "random").
        finder_params (dict or None): Parameters for the center-finding method (default: None).
        rbf_kernel (str): Type of RBF kernel to use (default: "gaussian").
        kernel_params (dict or None): Parameters for the RBF kernel (default: None).
        reg_lambda (float): Regularization parameter (default: 0.1).
        has_bias (bool): Whether to include a bias term in the model (default: False).
        seed (int): Random seed for reproducibility (default: 42).
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None

    def __init__(self, center_finder="random", finder_params=None,
                 rbf_kernel="gaussian", kernel_params=None,
                 reg_lambda=0.1, has_bias=False, seed=42):
        super().__init__()
        self.network = AdvancedRbfNet(center_finder, finder_params,
                                      rbf_kernel, kernel_params, reg_lambda, has_bias, seed)
        self.obj_scaler, self.loss_train, self.n_labels = None, None, None
        self.n_centers = None

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pred = self.network.predict(X)
        return self.obj_scaler.inverse_transform(pred)
    
    def predict_proba(self, X):
        return self.network.predict(X)

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)

    def __score_reg(self, X, y):
        y_pred = self.network.predict(X)
        return RegressionMetric().pearson_correlation_coefficient_square(y_true=y, y_pred=y_pred)

    def __scores_reg(self, X, y, list_metrics=("MSE", "MAE")):
        y_pred = self.network.predict(X)
        return self.__evaluate_reg(y_true=y, y_pred=y_pred, list_metrics=list_metrics)

    def __score_cls(self, X, y):
        y_pred = self.predict(X)
        return ClassificationMetric().accuracy_score(y_true=y, y_pred=y_pred)

    def __scores_cls(self, X, y, list_metrics=("AS", "RS")):
        list_errors = list(set(list_metrics) & set(self.CLS_OBJ_LOSSES))
        list_scores = list((set(self.SUPPORTED_CLS_METRICS.keys()) - set(self.CLS_OBJ_LOSSES)) & set(list_metrics))
        t1 = {}
        if len(list_errors) > 0:
            if self.n_labels > 2:
                y_pred = self.predict_proba(X)
            else:
                y_pred = self.predict(X)
            t1 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_errors)
        y_pred = self.predict(X)
        t2 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_scores)
        return {**t2, **t1}

    def evaluate(self, y_true, y_pred, list_metrics=None):
        pass

    def score(self, X, y):
        pass

    def scores(self, X, y, list_metrics=None):
        pass

    def save_loss_train(self, save_path="history", filename="loss.csv"):
        ## Save loss train to csv file
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} core doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        ## Save metrics to csv file
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        ## Save the predicted results to csv file
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="core.pkl"):
        ## Save core to pickle file
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="core.pkl"):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))


class AdvancedRbfRegressor(BaseAdvancedRbf, RegressorMixin):
    """
    This class implements an advanced Radial Basis Function (RBF) model tailored for regression problems.
    It extends the `BaseAdvancedRbf` and `RegressorMixin` classes, providing additional methods specific to regression tasks.

    Methods:
        __init__: Initializes the RBF regressor with specified configurations.
        fit: Fits the RBF network to the given training data and target values.
        score: Computes the R² score for the given test data and true values.
        scores: Computes multiple regression metrics for the given test data and true values.
        evaluate: Evaluates regression metrics for provided true and predicted values.

    Parameters for Initialization:
        center_finder (str): Method for finding RBF centers (default: "random").
        finder_params (dict or None): Parameters for the center-finding method (default: None).
        rbf_kernel (str): Type of RBF kernel to use (default: "gaussian").
        kernel_params (dict or None): Parameters for the RBF kernel (default: None).
        reg_lambda (float): Regularization parameter (default: 0.1).
        has_bias (bool): Whether to include a bias term in the model (default: False).
        seed (int): Random seed for reproducibility (default: 42).

    Usage:
        1. Instantiate the class with desired parameters.
        2. Use the `fit` method to train the model on input features `X` and targets `y`.
        3. Use `score`, `scores`, or `evaluate` to assess the model's performance.
    """

    def __init__(self, center_finder="random", finder_params=None,
                 rbf_kernel="gaussian", kernel_params=None,
                 reg_lambda=0.1, has_bias=False, seed=42):
        super().__init__(center_finder, finder_params, rbf_kernel, kernel_params, reg_lambda, has_bias, seed)

    def fit(self, X, y):
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
        self.obj_scaler = ObjectiveScaler(obj_name="self", ohe_scaler=None)
        y_scaled = self.obj_scaler.transform(y)
        self.network.fit(X, y_scaled)
        self.n_centers = self.network.n_centers
        return self

    def score(self, X, y):
        """
        Return the R2 squared on the given test data and y_true.
        """
        return self._BaseAdvancedRbf__score_reg(X, y)

    def scores(self, X, y, list_metrics=("MSE", "MAE")):
        """
        Return the list of regression metrics on the given test data and y_true.
        """
        return self._BaseAdvancedRbf__scores_reg(X, y, list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """
        Return the list of regression metrics on the given test data and y_true.
        """
        return self._BaseAdvancedRbf__evaluate_reg(y_true, y_pred, list_metrics)


class AdvancedRbfClassifier(BaseAdvancedRbf, ClassifierMixin):
    """
    This class implements an advanced Radial Basis Function (RBF) model specifically designed for
    classification tasks. It extends the `BaseAdvancedRbf` and `ClassifierMixin` classes, adding
    functionality for handling classification-specific metrics and data processing.

    Attributes:
        CLS_OBJ_LOSSES (list): A list of classification objective loss functions supported by the model, such as "CEL", "HL", "KLDL", and "BSL".
        n_labels (int): The number of unique labels in the target data.

    Methods:
        __init__: Initializes the RBF classifier with specified configurations.
        fit: Trains the RBF network using the input features `X` and target labels `y`.
        score: Computes the accuracy score for the given test data and true labels.
        scores: Computes a list of classification metrics for the given test data and true labels.
        evaluate: Evaluates classification metrics for the provided true and predicted labels.

    Parameters for Initialization:
        center_finder (str): Method for finding RBF centers (default: "random").
        finder_params (dict or None): Parameters for the center-finding method (default: None).
        rbf_kernel (str): Type of RBF kernel to use (default: "gaussian").
        kernel_params (dict or None): Parameters for the RBF kernel (default: None).
        reg_lambda (float): Regularization parameter (default: 0.1).
        has_bias (bool): Whether to include a bias term in the model (default: False).
        seed (int): Random seed for reproducibility (default: 42).

    Usage:
        1. Instantiate the class with desired parameters.
        2. Use the `fit` method to train the model on input features `X` and target labels `y`.
        3. Use `score`, `scores`, or `evaluate` to assess the model's performance.
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, center_finder="random", finder_params=None,
                 rbf_kernel="gaussian", kernel_params=None,
                 reg_lambda=0.1, has_bias=False, seed=42):
        super().__init__(center_finder, finder_params, rbf_kernel, kernel_params, reg_lambda, has_bias, seed)
        self.n_labels, self.classes_ = None, None

    def fit(self, X, y):
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
        self.obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        y_scaled = self.obj_scaler.transform(y)
        self.network.fit(X, y_scaled)
        self.n_centers = self.network.n_centers
        return self

    def score(self, X, y):
        """
        Return the accuracy score on the given test data and labels.
        """
        return self._BaseAdvancedRbf__score_cls(X, y)

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """
        Return the list of classification metrics on the given test data and labels.
        """
        return self._BaseAdvancedRbf__scores_cls(X, y, list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Return the list of classification metrics on the given test data and labels.
        """
        return self._BaseAdvancedRbf__evaluate_cls(y_true, y_pred, list_metrics)
