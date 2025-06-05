#!/usr/bin/env python
# Created by "Thieu" at 16:58, 10/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from evorbf import Data, NiaRbfClassifier
from sklearn.datasets import load_iris


## Load data object
# total classes = 3, total samples = 150, total features = 4
X, y = load_iris(return_X_y=True)
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
opt_paras = {"name": "WOA", "epoch": 50, "pop_size": 20}
print(NiaRbfClassifier.SUPPORTED_CLS_OBJECTIVES)
model = NiaRbfClassifier(size_hidden=25, center_finder="kmeans", regularization=False, obj_name="NPV",
                         optim="OriginalWOA", optim_params=opt_paras, verbose=True, seed=42,
                         lb=None, ub=None, mode='single', n_workers=None, termination=None)
## Train the model
model.fit(X=data.X_train, y=data.y_train)

## Test the model
y_pred = model.predict(data.X_test)
print(y_pred)

y_pred = model.predict_proba(data.X_test)
print(y_pred)

## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test))
print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["PS", "RS", "NPV", "F1S", "F2S"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["F2S", "CKS", "FBS"]))
