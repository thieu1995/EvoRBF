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
from mealpy import Problem, get_optimizer_by_name, Optimizer
from permetrics import ClassificationMetric, RegressionMetric
from evorbf.helpers.metrics import get_all_classification_metrics, get_all_regression_metrics
from evorbf.helpers import validator
from evorbf import RbfRegressor, RbfClassifier


class HyperparameterProblem(Problem):
    def __init__(self, bounds=None, minmax="max", X=None, y=None, model_class=None,
                 metric_class=None, obj_name=None, cv=5, seed=None, **kwargs):
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


class InaRbfTuner:

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()

    def __init__(self, problem_type="regression", bounds=None, cv=5, scoring="MSE",
                 optimizer="OriginalWOA", optimizer_paras=None, verbose=True, seed=None):
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
        self.optimizer_paras = optimizer_paras
        self.optimizer = self._set_optimizer(optimizer, optimizer_paras)
        self.best_params = None
        self.best_estimator = None
        self.loss_train = None

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

    def fit(self, X, y):
        self.problem = HyperparameterProblem(self.bounds, self.minmax, X, y, self.network_class, self.metric_class,
                                             obj_name=self.scoring, cv=self.cv, log_to=self.verbose)
        self.optimizer.solve(self.problem, seed=self.seed)
        self.best_params = self.optimizer.problem.decode_solution(self.optimizer.g_best.solution)
        self.best_estimator = self.network_class(**{**self.best_params, "seed": self.seed})
        self.best_estimator.fit(X, y)
        self.loss_train = self.optimizer.history.list_global_best_fit
        return self

    def predict(self, X):
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

    def save_performance_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
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
