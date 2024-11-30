#!/usr/bin/env python
# Created by "Thieu" at 21:45, 06/04/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_diabetes
from mealpy import StringVar, IntegerVar, FloatVar, BoolVar
from evorbf import Data, InaRbfTuner


## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1))

# Design the boundary (parameters)
my_bounds = [
    IntegerVar(lb=5, ub=21, name="size_hidden"),
    StringVar(valid_sets=("kmean", "random"), name="center_finder"),
    FloatVar(lb=(0.01,), ub=(3.0,), name="sigmas"),
    BoolVar(n_vars=1, name="regularization"),
    FloatVar(lb=(0.001, ), ub=(0.99, ), name="lamda"),
]

opt_paras = {"name": "WOA", "epoch": 10, "pop_size": 20}
model = InaRbfTuner(problem_type="regression", bounds=my_bounds, cv=3, scoring="MSE",
                      optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)
model.fit(data.X_train, data.y_train)
print(model.best_params)
print(model.best_estimator)
print(model.best_estimator.scores(data.X_test, data.y_test, list_metrics=("MSE", "RMSE", "MAPE", "NSE", "R2", "KGE")))
