#!/usr/bin/env python
# Created by "Thieu" at 16:12, 10/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from evorbf import Data, InaRbfClassifier
from sklearn.datasets import load_breast_cancer


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

## Create model
opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
print(InaRbfClassifier.SUPPORTED_CLS_OBJECTIVES)
model = InaRbfClassifier(size_hidden=25, center_finder="kmean", regularization=False, lamda=0.5, obj_name="AS",
                        optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)

## Train the model
model.fit(X=data.X_train, y=data.y_train)

## Test the model
y_pred = model.predict(data.X_test, return_prob=False)
print(y_pred)

y_pred = model.predict(data.X_test, return_prob=True)
print(y_pred)

## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test, method="AS"))
print(model.scores(X=data.X_test, y=data.y_test, list_methods=["PS", "RS", "NPV", "F1S", "F2S"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS"]))
