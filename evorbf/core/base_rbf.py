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
from mealpy import get_optimizer_by_class, Optimizer, get_all_optimizers, FloatVar
from sklearn.cluster import KMeans
from evorbf.helpers import validator
from evorbf.helpers.metrics import get_all_regression_metrics, get_all_classification_metrics


class CustomRBF:
    """Radial Basis Function

    This class defines the general RBF model that:
        + use non-linear Gaussian function
        + use inverse matrix multiplication instead of Gradient-based
        + set up regulation term with hyperparameter `lamda`

    Parameters
    ----------
    size_hidden : int, default=10
        The number of hidden nodes

    center_finder : str, default="kmeans"
        The method is used to find the cluster centers

    sigmas : float, int, np.ndarray, list, tuple, default=2.0
        The sigma values that are used in Gaussian function. In traditional RBF model, 1 sigma value is used
        for all of hidden nodes. But in Nature-inspired Algorithms (NIAs) based RBF model, each
        sigma is assigned to 1 hidden node.

    reg_lambda : float, default=0.1
        The lamda value is used in regularization term. If set to 0, then no L2 is applied

    seed : int, default=None
        The seed value is used for reproducibility.
    """
    def __init__(self, size_hidden=10, center_finder="kmeans", sigmas=2.0, reg_lambda=0.1, seed=None):
        self.size_hidden = size_hidden
        self.center_finder = center_finder
        self.sigmas = sigmas
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.centers, self.weights, self.weights_shape = None, None, None
        self.regularization = None

    def check_reg_lambda(self, reg_lambda):
        if type(reg_lambda) is float and reg_lambda > 0.0:
            return reg_lambda, True
        else:
            return reg_lambda, False

    def set_reg_lambda(self, reg_lambda):
        if type(reg_lambda) is float and reg_lambda > 0.0:
            self.reg_lambda = reg_lambda
            self.regularization = True
        else:
            self.reg_lambda, self.regularization = reg_lambda, False

    @staticmethod
    def calculate_centers(X, method="kmeans", n_clusters=5, seed=42):
        if method == "kmeans":
            kobj = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed).fit(X)
            return kobj.cluster_centers_
        elif method == "random":
            generator = np.random.default_rng(seed)
            return X[generator.choice(len(X), n_clusters, replace=False)]

    @staticmethod
    def calculate_rbf(X, c, sigma):
        # Calculate Radial Basis Function (Gaussian)
        # return np.exp(-np.sum((X - c)**2, axis=1) / (2 * sigmas**2))
        return np.exp(-np.linalg.norm(X - c, axis=1)**2 / (2 * sigma**2))

    def transform_X(self, X):
        # Calculate RBF layer outputs
        if self.centers is None:
            raise Exception("Model is not trained yet.")
        # Construct the RBF matrix
        rbf_layer = np.zeros((X.shape[0], self.size_hidden))
        for idx, c in enumerate(self.centers):
            rbf_layer[:, idx] = self.calculate_rbf(X, c, self.sigmas[idx])
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
        # Check regularization
        self.reg_lambda, self.regularization = self.check_reg_lambda(self.reg_lambda)
        # Check sigmas
        if isinstance(self.sigmas, (int, float, np.number)):
            self.sigmas = [self.sigmas, ] * self.size_hidden
        elif isinstance(self.sigmas, (list, tuple, np.ndarray)):
            if len(self.sigmas) != self.size_hidden:
                raise ValueError("sigmas must have equal length to size_hidden")
        else:
            raise ValueError("sigmas must be an number or list, tuple, or number array")
        # Initialize centers
        self.centers = self.calculate_centers(X, self.center_finder, self.size_hidden, self.seed)
        # Calculate RBF layer outputs
        rbf_layer = self.transform_X(X)
        if self.regularization:
            # Solve for weights using ridge regression (L2 regularization)
            lambda_iden = self.reg_lambda * np.eye(rbf_layer.shape[1])  # L2 regularization
            self.weights = np.linalg.inv(rbf_layer.T @ rbf_layer + lambda_iden) @ rbf_layer.T @ y
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
        This function is used for NIA-based RBF model. Whenever a solution is generated, it will call this function.
        """
        if self.centers is None:
            self.centers = self.calculate_centers(X, self.center_finder, self.size_hidden, self.seed)
        if self.regularization:
            self.sigmas = solution[:self.size_hidden]
            self.reg_lambda = solution[-1]
            rbf_layer = self.transform_X(X)
            # Solve for weights using ridge regression (L2 regularization)
            lambda_iden = self.reg_lambda * np.eye(rbf_layer.shape[1])  # L2 regularization
            self.weights = np.linalg.inv(rbf_layer.T @ rbf_layer + lambda_iden) @ rbf_layer.T @ y
        else:
            self.sigmas = solution
            rbf_layer = self.transform_X(X)
            # Solve for weights using pseudo-inverse
            self.weights = np.linalg.pinv(rbf_layer) @ y
    
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

    center_finder : str, default="kmeans"
        The method is used to find the cluster centers

    sigmas : float, default=2.0
        The sigma values that are used in Gaussian function. In traditional RBF model, 1 sigma value is used
        for all of hidden nodes. But in Nature-inspired Algorithms (NIAs) based RBF model, each
        sigma is assigned to 1 hidden node.

    reg_lambda : float, default=0.1
        The lamda value is used in regularization term. If set to 0, then no L2 is applied

    seed : int, default=None
        The seed value is used for reproducibility.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None

    def __init__(self, size_hidden=10, center_finder="kmeans", sigmas=2.0, reg_lambda=0.1, seed=None):
        super().__init__()
        self._net_class = CustomRBF
        self.size_hidden = size_hidden
        self.center_finder = center_finder
        self.sigmas = sigmas
        self.reg_lambda = reg_lambda
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

    def predict(self, X):
        """Predict the outcome of the feature X"""
        pred = self.network.predict(X)
        return self.obj_scaler.inverse_transform(pred)

    def predict_proba(self, X):
        """
        It is used for classification problem. The returned results are the probability for each sample
        """
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
        """Return the list of performance metrics of the prediction.
        You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics
        """
        pass

    def score(self, X, y):
        """Return the default metric of the prediction."""
        pass

    def scores(self, X, y, list_metrics=None):
        """Return the list of metrics of the prediction."""
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


class BaseNiaRbf(BaseRbf):
    """
    Defines the most general class for Nature-inspired Algorithm-based RBF models that inherits the BaseRbf class

    Note
    ----
        + In this model, the sigmas will be learned during the training process.
        + So the `sigmas` parameter is removed in the init function.
        + Besides, the `sigmas` is a list of value, each value represent a `sigma` for Gaussian function used in hidden node.

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

    optim_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    seed : int, default=None
        The seed value is used for reproducibility.

    Notes
    -----
    - This class is designed to be easily extended for hybrid metaheuristic-based RBF models.
    - Metrics can be customized using the Permetrics library: https://github.com/thieu1995/permetrics
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers(verbose=False).keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, size_hidden=10, center_finder="kmeans", regularization=True,
                 obj_name=None, optim="BaseGA", optim_paras=None, verbose=True, seed=None):
        super().__init__(size_hidden=size_hidden, center_finder=center_finder, seed=seed)
        self.regularization = regularization
        self.obj_name = obj_name
        self.optim_paras = optim_paras
        self.optimizer = self._set_optimizer(optim, optim_paras)
        self.verbose = verbose
        self.network, self.obj_scaler, self.obj_weights = None, None, None

    def _set_optimizer(self, optim=None, optim_paras=None):
        if type(optim) is str:
            opt_class = get_optimizer_by_class(optim)
            if type(optim_paras) is dict:
                return opt_class(**optim_paras)
            else:
                return opt_class(epoch=500, pop_size=50)
        elif isinstance(optim, Optimizer):
            if type(optim_paras) is dict:
                return optim.set_parameters(optim_paras)
            return optim
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def _get_history_loss(self, optim=None):
        list_global_best = optim.history.list_global_best
        # 2D array / matrix 2D
        return np.array([agent.target.fitness for agent in list_global_best])

    def objective_function(self, solution=None):
        pass

    def fit(self, X, y, lb=None, ub=None, save_population=False):
        self.network, self.obj_scaler = self.create_network(X, y)
        y_scaled = self.obj_scaler.transform(y)
        self.X_temp, self.y_temp = X, y_scaled
        if self.regularization:
            problem_size = self.size_hidden + 1
        else:
            problem_size = self.size_hidden
        if lb is None or ub is None:
            ub = [np.mean(np.max(X, axis=0)), ] * problem_size
            lb = [0.001, ] * problem_size
        elif type(lb) in (list, tuple, np.ndarray) and type(ub) in (list, tuple, np.ndarray):
            if len(lb) == len(ub):
                if len(lb) == 1:
                    lb = lb * problem_size
                    ub = ub * problem_size
                elif len(lb) != problem_size:
                    raise ValueError(f"Invalid lb and ub. Their length should equal to 1 or {problem_size}.")
            else:
                raise ValueError(f"Invalid lb and ub. They should have the same length.")
        elif type(lb) in (int, float) and type(ub) in (int, float):
            lb = [lb, ] * problem_size
            ub = [ub, ] * problem_size
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
        self.loss_train = self._get_history_loss(optim=self.optimizer)
        return self
