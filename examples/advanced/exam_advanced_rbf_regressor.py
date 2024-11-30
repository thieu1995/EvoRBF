#!/usr/bin/env python
# Created by "Thieu" at 22:22, 29/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from evorbf import Data, AdvancedRbfRegressor
from sklearn.datasets import load_diabetes


def prepare_data():
    ## Load data object
    # total samples = 442, total features = 10
    X, y = load_diabetes(return_X_y=True)
    data = Data(X, y)

    ## Split train and test
    data.split_train_test(test_size=0.2, random_state=2, inplace=True)
    print(data.X_train.shape, data.X_test.shape)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
    data.X_test = scaler_X.transform(data.X_test)

    data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
    data.y_test = scaler_y.transform(data.y_test)

    print(type(data.X_train), type(data.y_train))
    return data, scaler_X, scaler_y


data, scaler_X, scaler_y = prepare_data()

model = AdvancedRbfRegressor(center_finder="random", finder_params={"n_centers": 15},
                 rbf_kernel="gaussian", kernel_params={"sigma": 1.5},
                 reg_lambda=0.1, has_bias=False, seed=42)

model = AdvancedRbfRegressor(center_finder="random", finder_params=None,        # Default n_centers = 10
                 rbf_kernel="gaussian", kernel_params=None,                     # Default sigma = 1.0
                 reg_lambda=0.1, has_bias=False, seed=42)

model = AdvancedRbfRegressor(center_finder="kmeans", finder_params={"n_centers": 20},
                 rbf_kernel="multiquadric", kernel_params=None,
                 reg_lambda=0.1, has_bias=False, seed=42)

model = AdvancedRbfRegressor(center_finder="meanshift", finder_params={"bandwidth": 0.45},      # Give us 11 hidden nodes
                 rbf_kernel="inverse_multiquadric", kernel_params={"sigma": 1.5},
                 reg_lambda=0.5, has_bias=True, seed=42)

model = AdvancedRbfRegressor(center_finder="dbscan", finder_params={"eps": 0.15},      # Give us 13 hidden nodes
                 rbf_kernel="multiquadric", kernel_params={"sigma": 1.5},
                 reg_lambda=0.5, has_bias=True, seed=42)

model = AdvancedRbfRegressor(center_finder="dbscan", finder_params={"eps": 0.14},      # Give us 9 hidden nodes
                 rbf_kernel="multiquadric", kernel_params={"sigma": 1.5},
                 reg_lambda=None, has_bias=False, seed=42)

## Train the model
model.fit(data.X_train, data.y_train)
print(model.n_centers)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)

## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test))
print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["R2", "NSE", "MAPE"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["R2", "NSE", "MAPE", "NNSE"]))
