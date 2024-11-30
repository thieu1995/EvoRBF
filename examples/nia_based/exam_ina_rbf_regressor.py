#!/usr/bin/env python
# Created by "Thieu" at 17:06, 22/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.linear_model import LinearRegression
from evorbf import Data, NiaRbfRegressor
from sklearn.datasets import load_diabetes


## Load data object
# total samples = 442, total features = 10
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", ))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", ))
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))


opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 20}
model = NiaRbfRegressor(size_hidden=25, center_finder="kmeans", regularization=False, obj_name="MSE",
                        optim="BaseGA", optim_paras=opt_paras, verbose=True, seed=42)

## Train the model
model.fit(data.X_train, data.y_train)

## Test the model
y_pred = model.predict(data.X_test)

print(model.optimizer.g_best.solution)
## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test))
print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["R2", "R", "KGE", "MAPE"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["MSE", "RMSE", "R2S", "NSE", "KGE", "MAPE"]))


## Test Linear Regression
model = LinearRegression()
model.fit(data.X_train, data.y_train)
y_pred = model.predict(data.X_test)

from permetrics import RegressionMetric
metric = RegressionMetric(data.y_test, y_pred)
print(metric.get_metrics_by_list_names(["MSE", "RMSE", "R2S", "NSE", "KGE", "MAPE"]))
