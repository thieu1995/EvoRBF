#!/usr/bin/env python
# Created by "Thieu" at 21:45, 06/04/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_diabetes
from mealpy import StringVar, IntegerVar, FloatVar
from evorbf import Data, NiaRbfTuner


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
    StringVar(valid_sets=("kmeans", "random"), name="center_finder"),
    FloatVar(lb=(0.01,), ub=(3.0,), name="sigmas"),
    FloatVar(lb=(0, ), ub=(1.0, ), name="reg_lambda"),
]

opt_paras = {"name": "WOA", "epoch": 10, "pop_size": 20}
model = NiaRbfTuner(problem_type="regression", bounds=my_bounds, cv=3, scoring="MSE",
                    optim="OriginalWOA", optim_params=opt_paras, verbose=True, seed=42,
                    mode='single', n_workers=None, termination=None)

model.fit(data.X_train, data.y_train)
print(model.best_params)
print(model.best_estimator)
print(model.best_estimator.scores(data.X_test, data.y_test, list_metrics=("MSE", "RMSE", "MAPE", "NSE", "R2", "KGE")))
