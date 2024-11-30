#!/usr/bin/env python
# Created by "Thieu" at 10:31, 30/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from evorbf import Data, AdvancedRbfClassifier
from sklearn.datasets import load_breast_cancer


def prepare_data():
    ## Load data object
    # total classes = 2, total samples = 569, total features = 30
    X, y = load_breast_cancer(return_X_y=True)
    data = Data(X, y)

    ## Split train and test
    data.split_train_test(test_size=0.2, random_state=2, inplace=True, shuffle=True)
    print(data.X_train.shape, data.X_test.shape)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
    data.X_test = scaler_X.transform(data.X_test)

    data.y_train, scaler_y = data.encode_label(data.y_train)
    data.y_test = scaler_y.transform(data.y_test)

    print(type(data.X_train))
    print(data.y_train.shape)
    return data, scaler_X, scaler_y


data, scaler_X, scaler_y = prepare_data()

model = AdvancedRbfClassifier(center_finder="random", finder_params={"n_centers": 15},
                 rbf_kernel="gaussian", kernel_params={"sigma": 1.5},
                 reg_lambda=0.1, has_bias=False, seed=42)

model = AdvancedRbfClassifier(center_finder="random", finder_params=None,        # Default n_centers = 10
                 rbf_kernel="gaussian", kernel_params=None,                     # Default sigma = 1.0
                 reg_lambda=0.1, has_bias=False, seed=42)

model = AdvancedRbfClassifier(center_finder="kmeans", finder_params={"n_centers": 20},
                 rbf_kernel="multiquadric", kernel_params=None,
                 reg_lambda=0.1, has_bias=False, seed=42)

model = AdvancedRbfClassifier(center_finder="meanshift", finder_params={"bandwidth": 0.6},      # Give us 28 hidden nodes
                 rbf_kernel="inverse_multiquadric", kernel_params={"sigma": 1.5},
                 reg_lambda=0.5, has_bias=True, seed=42)

model = AdvancedRbfClassifier(center_finder="dbscan", finder_params={"eps": 0.2},      # Give us 42 hidden nodes
                 rbf_kernel="multiquadric", kernel_params={"sigma": 1.5},
                 reg_lambda=0.5, has_bias=True, seed=42)

model = AdvancedRbfClassifier(center_finder="dbscan", finder_params={"eps": 0.175},      # Give us 16 hidden nodes
                 rbf_kernel="multiquadric", kernel_params={"sigma": 1.5},
                 reg_lambda=None, has_bias=False, seed=42)

## Train the model
model.fit(data.X_train, data.y_train)
print(model.n_centers)

## Test the model
y_pred = model.predict_proba(data.X_test)
print(y_pred)
print(model.predict(data.X_test))

## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test))
print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["PS", "RS", "NPV", "F1S", "F2S"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS"]))
