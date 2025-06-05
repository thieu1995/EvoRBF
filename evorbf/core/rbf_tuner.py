#!/usr/bin/env python
# Created by "Thieu" at 11:11, 11/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from mealpy import Problem, get_optimizer_by_class, Optimizer
from permetrics import ClassificationMetric, RegressionMetric
from evorbf.helpers.metrics import get_all_classification_metrics, get_all_regression_metrics
from evorbf.helpers import validator
from evorbf import RbfRegressor, RbfClassifier


class HyperparameterProblem(Problem):
    """
    A class for defining a hyperparameter optimization problem using cross-validation.

    Inherits from the `Problem` class and implements an objective function to evaluate
    a given set of hyperparameters by training and evaluating a specified model class
    using cross-validation.

    Attributes:
        model_class (type): The class of the model to be optimized.
        model (object): The model instance created during optimization.
        X (array-like): The feature matrix for training the model.
        y (array-like): The target variable for training the model.
        metric_class (type): The class used for evaluating the model's performance.
        obj_name (str): The name of the metric to be extracted from the metric class.
        cv (int): The number of folds in cross-validation (default is 5).
        seed (int): The seed for random number generation for reproducibility.
        kf (KFold): The cross-validation iterator.

    Methods:
        obj_func(x):
            Evaluates the performance of the model with given hyperparameters using
            cross-validation and returns the mean performance score.
    """
    def __init__(self, bounds=None, minmax="max", X=None, y=None, model_class=None,
                 metric_class=None, obj_name=None, cv=5, seed=None, **kwargs):
        """
        Initializes the HyperparameterProblem with the given bounds and other attributes.

        Parameters:
            bounds (list of tuples): The search space for hyperparameters, each tuple
                                     represents the min and max values for a parameter.
            minmax (str): Indicates whether the objective function should be maximized
                          ("max") or minimized ("min").
            X (array-like): The feature matrix for training the model.
            y (array-like): The target variable for training the model.
            model_class (type): The class of the model to be optimized.
            metric_class (type): The class used for evaluating the model's performance.
            obj_name (str): The specific metric name to be used for model evaluation.
            cv (int): The number of folds for cross-validation (default is 5).
            seed (int): The seed for random number generation to ensure reproducibility.
            **kwargs: Additional keyword arguments passed to the parent class's initializer.
        """
        self.model_class = model_class
        self.model = None
        self.X = X
        self.y = y
        self.metric_class = metric_class
        self.obj_name = obj_name
        self.cv = cv
        self.seed = seed
        self.kf = KFold(n_splits=cv, shuffle=True, random_state=self.seed)
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        """
        Evaluates the model's performance for a given set of hyperparameters using
        cross-validation and returns the mean performance score.

        Parameters:
            x (array-like): The hyperparameter configuration to be evaluated.

        Returns:
            float: The mean score of the model across all cross-validation folds.
        """
        x_decoded = self.decode_solution(x)
        self.model = self.model_class(**x_decoded)
        scores = []
        # Perform custom cross-validation
        for train_idx, test_idx in self.kf.split(self.X):
            # Split the data into training and test sets
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            # Train the model on the training set
            self.model.fit(X_train, y_train)
            # Make predictions on the test set
            y_pred = self.model.predict(X_test)
            # Calculate accuracy for the current fold
            mt = self.metric_class(y_test, y_pred)
            score = mt.get_metric_by_name(self.obj_name)[self.obj_name]
            # Accumulate accuracy across folds
            scores.append(score)
        return np.mean(scores)


class NiaRbfTuner:
    """
    A class for tuning hyperparameters of Radial Basis Function (RBF) neural networks using optimization algorithms.

    This class supports both regression and classification tasks and can be configured to use different optimization
    algorithms from the Mealpy library to find the best set of hyperparameters for an RBF model.

    Attributes:
        SUPPORTED_CLS_METRICS (dict): Dictionary of supported metrics for classification tasks.
        SUPPORTED_REG_METRICS (dict): Dictionary of supported metrics for regression tasks.
        network_class (type): The class of the RBF network, either `RbfRegressor` or `RbfClassifier`.
        scoring (str): The metric used for evaluating the model performance.
        minmax (str): Indicates whether the scoring metric should be maximized or minimized.
        metric_class (type): The class for the metric used for model evaluation, either `RegressionMetric` or `ClassificationMetric`.
        seed (int or None): Random seed for reproducibility.
        problem_type (str): Indicates whether the problem is a "regression" or "classification" task.
        bounds (list of tuples): The search space for hyperparameters.
        cv (int): The number of folds in cross-validation.
        verbose (str): Level of verbosity for the optimization process.
        optim_params (dict or None): Parameters for the optimization algorithm.
        optimizer (Optimizer): The optimization algorithm used for tuning hyperparameters.
        best_params (dict or None): The best hyperparameters found.
        best_estimator (object or None): The trained model with the best hyperparameters.
        loss_train (list or None): The training loss history during the optimization process.

    Methods:
        _set_optim(optim, optim_params):
            Initializes and returns the optimizer instance based on the provided algorithm name and parameters.
        fit(X, y):
            Fits the model to the training data and optimizes the hyperparameters using the specified optimizer.
        predict(X):
            Predicts the target values for the input data using the trained model.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()

    def __init__(self, problem_type="regression", bounds=None, cv=5, scoring="MSE",
                 optim="OriginalWOA", optim_params=None, verbose=True, seed=None):
        """
        Initializes the NiaRbfTuner with the specified parameters and settings.

        Parameters:
            problem_type (str): Type of problem, "regression" or "classification" (default is "regression").
            bounds (list of tuples): The search space for hyperparameters (default is None).
            cv (int): The number of folds for cross-validation (default is 5).
            scoring (str): The metric used for evaluating the model's performance (default is "MSE").
            optim (str): The name of the optimization algorithm (default is "OriginalWOA").
            optim_params (dict or None): Parameters for the optimization algorithm (default is None).
            verbose (bool): If True, displays console output during optimization (default is True).
            seed (int or None): Random seed for reproducibility (default is None).
        """
        if problem_type == "regression":
            self.network_class = RbfRegressor
            self.scoring = validator.check_str("scoring", scoring, self.SUPPORTED_REG_METRICS)
            self.minmax = self.SUPPORTED_REG_METRICS[self.scoring]
            self.metric_class = RegressionMetric
        else:
            self.network_class = RbfClassifier
            self.scoring = validator.check_str("scoring", scoring, self.SUPPORTED_CLS_METRICS)
            self.minmax = self.SUPPORTED_CLS_METRICS[self.scoring]
            self.metric_class = ClassificationMetric
        self.seed = seed
        self.problem_type = problem_type
        self.bounds = bounds
        self.cv = cv
        self.verbose = "console" if verbose else "None"
        self.optim_params = optim_params
        self.optimizer = self._set_optim(optim, optim_params)
        self.best_params = None
        self.best_estimator = None
        self.loss_train = None

    def _set_optim(self, optim=None, optim_params=None):
        """
        Sets up the optimizer instance based on the algorithm name and parameters.

        Parameters:
            optim (str or Optimizer): The name of the optimizer as a string or an instance of an optimizer.
            optim_params (dict): Parameters for the optimizer.

        Returns:
            Optimizer: An initialized optimizer instance.

        Raises:
            TypeError: If `optim` is neither a string nor an `Optimizer` instance.
        """
        if type(optim) is str:
            opt_class = get_optimizer_by_class(optim)
            if type(optim_params) is dict:
                return opt_class(**optim_params)
            else:
                return opt_class(epoch=500, pop_size=50)
        elif isinstance(optim, Optimizer):
            if type(optim_params) is dict:
                return optim.set_parameters(optim_params)
            return optim
        else:
            raise TypeError(f"optim needs to set as a string and supported by Mealpy library.")

    def fit(self, X, y):
        """
        Fits the RBF model to the training data and optimizes the hyperparameters.

        Parameters:
            X (array-like): The feature matrix for training.
            y (array-like): The target variable for training.

        Returns:
            self: The instance of the fitted `NiaRbfTuner` with optimized hyperparameters.
        """
        self.problem = HyperparameterProblem(self.bounds, self.minmax, X, y,
                                             self.network_class, self.metric_class,
                                             obj_name=self.scoring, cv=self.cv, log_to=self.verbose)
        self.optimizer.solve(self.problem, seed=self.seed)
        self.best_params = self.optimizer.problem.decode_solution(self.optimizer.g_best.solution)
        self.best_estimator = self.network_class(**{**self.best_params, "seed": self.seed})
        self.best_estimator.fit(X, y)
        self.loss_train = self.optimizer.history.list_global_best_fit
        return self

    def predict(self, X):
        """
        Predicts the target values for the input data using the trained model.

        Parameters:
            X (array-like): The feature matrix for making predictions.

        Returns:
            array-like: Predicted target values.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.best_params is None or self.best_estimator is None:
            raise ValueError(f"Model is not trained, please call the fit() function.")
        return self.best_estimator.predict(X)

    def save_convergence(self, save_path="history", filename="convergence.csv"):
        """
        Save the convergence (fitness value) during the training process to csv file.

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} network doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_performance_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"),
                                 save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to csv file

        Parameters
        ----------
        y_true : ground truth data
        y_pred : predicted output
        list_metrics : list of evaluation metrics
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.best_estimator.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save the predicted results to csv file

        Parameters
        ----------
        X : The features data, nd.ndarray
        y_true : The ground truth data
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="network.pkl"):
        """
        Save network to pickle file

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".pkl" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="network.pkl"):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))
