#!/usr/bin/env python
# Created by "Thieu" at 09:48, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import BaseEstimator
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar
from sklearn.cluster import KMeans
from evorbf.helpers import validator
from evorbf.helpers.metrics import get_all_regression_metrics, get_all_classification_metrics


class RBF:
    """Radial Basis Function

    This class defines the general RBF model that:
        + use non-linear Gaussian function
        + use inverse matrix multiplication instead of Gradient-based
        + set up regulation term with hyperparameter `lamda`

    Parameters
    ----------
    size_hidden : int, default=10
        The number of hidden nodes

    center_finder : str, default="kmean"
        The method is used to find the cluster centers

    sigmas : float, default=2.0
        The sigma values that are used in Gaussian function. In traditional RBF model, 1 sigma value is used
        for all of hidden nodes. But in Intelligence Nature-inspired Algorithms (INAs) based RBF model, each
        sigma is assigned to 1 hidden node.

    regularization : bool, default=False
        Set up the regularization or not

    lamda : float, default=0.01
        The lamda value is used in regularization term

    seed : int, default=None
        The seed value is used for reproducibility.
    """
    def __init__(self, size_hidden=10, center_finder="kmean", sigmas=2.0, regularization=False, lamda=0.01, seed=None):
        self.size_hidden = size_hidden
        self.center_finder = center_finder
        self.sigmas = sigmas
        self.regularization = regularization
        self.lamda = lamda
        self.seed = seed
        self.centers, self.weights, self.weights_shape = None, None, None

    @staticmethod
    def calculate_centers(X, method="kmean", n_clusters=5, seed=42):
        if method == "kmean":
            kobj = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(X)
            return kobj.cluster_centers_
        elif method == "random":
            generator = np.random.default_rng(seed)
            return X[generator.choice(len(X), n_clusters, replace=False)]

    @staticmethod
    def calculate_rbf(X, c, sigmas):
        # Calculate Radial Basis Function (Gaussian)
        return np.exp(-np.sum((X - c)**2, axis=1) / (2 * sigmas**2))

    def transform_X(self, X):
        # Calculate RBF layer outputs
        if self.centers is None:
            raise Exception("Model is not trained yet.")
        rbf_layer = np.zeros((X.shape[0], self.size_hidden))
        for i in range(X.shape[0]):
            rbf_layer[i] = self.calculate_rbf(X[i], self.centers, self.sigmas)
        return rbf_layer

    def fit(self, X, y):
        """Fit the core to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
            Returns a trained RBF core.
        """
        # Initialize centers
        self.centers = self.calculate_centers(X, self.center_finder, self.size_hidden, self.seed)
        # Calculate RBF layer outputs
        rbf_layer = self.transform_X(X)
        if self.regularization:
            # Solve for weights using ridge regression (L2 regularization)
            iden = np.identity(self.size_hidden)
            self.weights = np.linalg.inv(rbf_layer.T @ rbf_layer + self.lamda * iden) @ rbf_layer.T @ y
        else:
            # Solve for weights using pseudo-inverse
            self.weights = np.linalg.pinv(rbf_layer) @ y
        return self

    def predict(self, X):
        """Predict using the Radial Basis Function core.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        rbf_layer = self.transform_X(X)
        return rbf_layer @ self.weights
    
    def update_weights_from_solution(self, solution, X, y):
        """
        This function is used for INA-based RBF model. Whenever a solution is generated, it will call this function.
        """
        if self.centers is None:
            self.centers = self.calculate_centers(X, self.center_finder, self.size_hidden, self.seed)
        if self.weights_shape is None:
            if y.ndim == 1:
                self.weights_shape = (self.size_hidden, 1)
            else:
                self.weights_shape = (self.size_hidden, y.shape[1])
        self.sigmas = solution[:self.size_hidden]
        self.weights = np.reshape(solution[self.size_hidden:], self.weights_shape)
    
    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights_size(self):
        return self.weights.size()


class BaseRbf(BaseEstimator):
    """
    Defines the most general class for RBF network that inherits the BaseEstimator class of Scikit-Learn library.

    Parameters
    ----------
    size_hidden : int, default=10
        The number of hidden nodes

    center_finder : str, default="kmean"
        The method is used to find the cluster centers

    sigmas : float, default=2.0
        The sigma values that are used in Gaussian function. In traditional RBF model, 1 sigma value is used
        for all of hidden nodes. But in Intelligence Nature-inspired Algorithms (INAs) based RBF model, each
        sigma is assigned to 1 hidden node.

    regularization : bool, default=False
        Set up the regularization or not

    lamda : float, default=0.01
        The lamda value is used in regularization term

    seed : int, default=None
        The seed value is used for reproducibility.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None

    def __init__(self, size_hidden=10, center_finder="kmean", sigmas=2.0, regularization=False, lamda=0.01, seed=None):
        super().__init__()
        self._net_class = RBF
        self.size_hidden = size_hidden
        self.center_finder = center_finder
        self.sigmas = sigmas
        self.regularization = regularization
        self.lamda = lamda
        self.seed = seed
        self.parameters = {}
        self.network, self.obj_scaler, self.loss_train, self.n_labels = None, None, None, None

    @staticmethod
    def _check_method(method=None, list_supported_methods=None) -> str:
        if type(method) is str:
            return validator.check_str("method", method, list_supported_methods)
        else:
            raise ValueError(f"method should be a string and belongs to {list_supported_methods}")

    def create_network(self, X, y):
        return None, None

    def fit(self, X, y):
        self.network, self.obj_scaler = self.create_network(X, y)
        y_scaled = self.obj_scaler.transform(y)
        self.network.fit(X, y_scaled)
        return self

    def predict(self, X, return_prob=False):
        """
        Inherit the predict function from BaseRbf class, with 1 more parameter `return_prob`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        return_prob : bool, default=False
            It is used for classification problem:

            - If True, the returned results are the probability for each sample
            - If False, the returned results are the predicted labels
        """
        pred = self.network.predict(X)
        if return_prob:
            return pred
        return self.obj_scaler.inverse_transform(pred)

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)
    
    def __score_reg(self, X, y, method="RMSE"):
        method = self._check_method(method, list(self.SUPPORTED_REG_METRICS.keys()))
        y_pred = self.network.predict(X)
        return RegressionMetric(y, y_pred).get_metric_by_name(method)[method]
    
    def __scores_reg(self, X, y, list_methods=("MSE", "MAE")):
        y_pred = self.network.predict(X)
        return self.__evaluate_reg(y_true=y, y_pred=y_pred, list_metrics=list_methods)

    def __score_cls(self, X, y, method="AS"):
        method = self._check_method(method, list(self.SUPPORTED_CLS_METRICS.keys()))
        return_prob = False
        if self.n_labels > 2:
            if method in self.CLS_OBJ_LOSSES:
                return_prob = True
        y_pred = self.predict(X, return_prob=return_prob)
        cm = ClassificationMetric(y_true=y, y_pred=y_pred)
        return cm.get_metric_by_name(method)[method]

    def __scores_cls(self, X, y, list_methods=("AS", "RS")):
        list_errors = list(set(list_methods) & set(self.CLS_OBJ_LOSSES))
        list_scores = list((set(self.SUPPORTED_CLS_METRICS.keys()) - set(self.CLS_OBJ_LOSSES)) & set(list_methods))
        t1 = {}
        if len(list_errors) > 0:
            return_prob = False
            if self.n_labels > 2:
                return_prob = True
            y_pred = self.predict(X, return_prob=return_prob)
            t1 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_errors)
        y_pred = self.predict(X, return_prob=False)
        t2 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_scores)
        return {**t2, **t1}

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def score(self, X, y, method=None):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        pass

    def scores(self, X, y, list_methods=None):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
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
        y_pred = self.predict(X, return_prob=False)
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


class BaseInaRbf(BaseRbf):
    """
    Defines the most general class for Intelligence Nature-inspired Algorithm-based RBF models that inherits the BaseRbf class

    Note:
    -----
        + In this models, the sigmas will be learned during the training process.
        + So the `sigmas` parameter is removed in the init function.
        + Besides, the `sigmas` is a list of value, each value represent a `sigma` for Gaussian function used in hidden node.

    Parameters
    ----------
    size_hidden : int, default=10
        The number of hidden nodes

    center_finder : str, default="kmean"
        The method is used to find the cluster centers

    regularization : bool, default=False
        Set up the regularization or not

    lamda : float, default=0.01
        The lamda value is used in regularization term

    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    seed : int, default=None
        The seed value is used for reproducibility.
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers().keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, size_hidden=10, center_finder="kmean", regularization=False, lamda=0.01,
                 obj_name=None, optimizer="BaseGA", optimizer_paras=None, verbose=True, seed=None):
        super().__init__(size_hidden=size_hidden, center_finder=center_finder,
                         regularization=regularization, lamda=lamda, seed=seed)
        self.obj_name = obj_name
        self.optimizer_paras = optimizer_paras
        self.optimizer = self._set_optimizer(optimizer, optimizer_paras)
        self.verbose = verbose
        self.network, self.obj_scaler, self.obj_weights = None, None, None

    def _set_optimizer(self, optimizer=None, optimizer_paras=None):
        if type(optimizer) is str:
            opt_class = get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                return opt_class(**optimizer_paras)
            else:
                return opt_class(epoch=500, pop_size=50)
        elif isinstance(optimizer, Optimizer):
            if type(optimizer_paras) is dict:
                return optimizer.set_parameters(optimizer_paras)
            return optimizer
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def _get_history_loss(self, optimizer=None):
        list_global_best = optimizer.history.list_global_best
        # 2D array / matrix 2D
        return np.array([agent.target.fitness for agent in list_global_best])

    def objective_function(self, solution=None):
        pass

    def fit(self, X, y, lb=(-1.0, ), ub=(1.0, ), save_population=False):
        self.network, self.obj_scaler = self.create_network(X, y)
        y_scaled = self.obj_scaler.transform(y)
        self.X_temp, self.y_temp = X, y_scaled
        if y_scaled.ndim == 1:
            problem_size = self.size_hidden
        else:
            problem_size = self.size_hidden * y_scaled.shape[1]
        ub_sigma = [np.mean(np.max(X, axis=0)), ] * self.size_hidden
        lb_sigma = [0.01, ] * self.size_hidden
        if type(lb) in (list, tuple, np.ndarray) and type(ub) in (list, tuple, np.ndarray):
            if len(lb) == len(ub):
                if len(lb) == 1:
                    lb = np.array(lb_sigma + list(lb) * problem_size, dtype=float)
                    ub = np.array(ub_sigma + list(ub) * problem_size, dtype=float)
                elif len(lb) != problem_size:
                    raise ValueError(f"Invalid lb and ub. Their length should be equal to 1 or problem_size.")
            else:
                raise ValueError(f"Invalid lb and ub. They should have the same length.")
        elif type(lb) in (int, float) and type(ub) in (int, float):
            lb = lb_sigma + [float(lb), ] * problem_size
            ub = ub_sigma + [float(ub), ] * problem_size
        else:
            raise ValueError(f"Invalid lb and ub. They should be a number of list/tuple/np.ndarray with size equal to problem_size")
        log_to = "console" if self.verbose else "None"
        if self.obj_name is None:
            raise ValueError("obj_name can't be None")
        else:
            if self.obj_name in self.SUPPORTED_REG_OBJECTIVES.keys():
                minmax = self.SUPPORTED_REG_OBJECTIVES[self.obj_name]
            elif self.obj_name in self.SUPPORTED_CLS_OBJECTIVES.keys():
                minmax = self.SUPPORTED_CLS_OBJECTIVES[self.obj_name]
            else:
                raise ValueError("obj_name is not supported. Please check the library: permetrics to see the supported objective function.")
        problem = {
            "obj_func": self.objective_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": minmax,
            "log_to": log_to,
            "save_population": save_population,
            "obj_weights": self.obj_weights
        }
        self.optimizer.solve(problem, seed=self.seed)
        self.network.update_weights_from_solution(self.optimizer.g_best.solution, X, y_scaled)
        self.loss_train = self._get_history_loss(optimizer=self.optimizer)
        return self
