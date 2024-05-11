
<p align="center">
<img style="max-width:100%;" src="https://thieu1995.github.io/post/2023-08/evorbf1.png" alt="EvoRBF"/>
</p>

---


[![GitHub release](https://img.shields.io/badge/release-1.0.0-yellow.svg)](https://github.com/thieu1995/evorbf/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/evorbf) 
[![PyPI version](https://badge.fury.io/py/evorbf.svg)](https://badge.fury.io/py/evorbf)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/evorbf.svg)
![PyPI - Status](https://img.shields.io/pypi/status/evorbf.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/evorbf.svg)
[![Downloads](https://static.pepy.tech/badge/evorbf)](https://pepy.tech/project/evorbf)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/evorbf/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/evorbf/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/evorbf.svg)
[![Documentation Status](https://readthedocs.org/projects/evorbf/badge/?version=latest)](https://evorbf.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/evorbf.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11136007.svg)](https://doi.org/10.5281/zenodo.11136007)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


**EvoRBF** is a Python library that implements a framework 
for training Radial Basis Function (RBF) networks using `Intelligence Nature-inspired Algorithms (INAs)`. It provides a 
comparable alternative to the traditional RBF network and is compatible with the Scikit-Learn library. With EvoRBF, you can 
perform searches and hyperparameter tuning using the functionalities provided by the Scikit-Learn library.

| **EvoRBF**                   | **Evolving Radial Basis Function Network**                                  |
|------------------------------|-----------------------------------------------------------------------------|
| **Free software**            | GNU General Public License (GPL) V3 license                                 |
| **Provided Estimator**       | RbfRegressor, RbfClassifier, InaRbfRegressor, InaRbfClassifier, InaRbfTuner |
| **Provided ML models**       | \> 400 Models                                                               |
| **Supported metrics**        | \>= 67 (47 regressions and 20 classifications)                              |
| **Supported loss functions** | \>= 61 (45 regressions and 16 classifications)                              |
| **Documentation**            | https://evorbf.readthedocs.io                                               | 
| **Python versions**          | \>= 3.8.x                                                                   |  
| **Dependencies**             | numpy, scipy, scikit-learn, pandas, mealpy, permetrics                      |


# Citation Request 

If you want to understand how Intelligence Nature-inspired Algorithms is applied to Radial Basis Function Network, you 
need to read the paper titled "Application of artificial intelligence in estimating mining capital expenditure using radial basis function neural network optimized by metaheuristic algorithms". 
The paper can be accessed at the following [this link](https://doi.org/10.1016/B978-0-443-18764-3.00015-1)


```bibtex
@software{thieu_2024_11136008,
  author       = {Nguyen Van Thieu},
  title        = {EvoRBF: Evolving Radial Basis Function Network by Intelligent Nature-inspired Algorithms},
  month        = may,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.11136007},
  url          = {https://doi.org/10.5281/zenodo.11136007}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}
```


# Usage

* Install the [current PyPI release](https://pypi.python.org/pypi/evorbf):
```sh 
$ pip install evorbf
```

After installation, you can check EvoRBF version:

```sh
$ python
>>> import evorbf
>>> evorbf.__version__
```

In this example below, we will use Whale Optimization Algorithm to optimize the `sigmas` (in non-linear Gaussian 
kernel) and `weights` (of hidden-output layer) in RBF network (WOA-RBF model) for Diabetes prediction problem.

```python
import numpy as np
from evorbf import Data, InaRbfRegressor
from sklearn.datasets import load_diabetes

## Load data object
# total samples = 442, total features = 10
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", ))
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))

## Create model
opt_paras = {"name": "WOA", "epoch": 500, "pop_size": 20}
model = InaRbfRegressor(size_hidden=25, center_finder="kmean", regularization=False, lamda=0.5, obj_name="MSE",
                        optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True, seed=42)

## Train the model
model.fit(data.X_train, data.y_train, lb=-1., ub=2.)

## Test the model
y_pred = model.predict(data.X_test)

print(model.optimizer.g_best.solution)
## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test, method="RMSE"))
print(model.scores(X=data.X_test, y=data.y_test, list_methods=["R2", "R", "KGE", "MAPE"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["MSE", "RMSE", "R2S", "NSE", "KGE", "MAPE"]))
```

Please go check out the [examples](/examples) folder. You'll be surprised by what this library can do for your problem.
You can also read the [documentation](https://evorbf.readthedocs.io/) for more detailed installation 
instructions, explanations, and examples.


### Official Links (Get support for questions and answers)

* [Official source code repository](https://github.com/thieu1995/evorbf)
* [Official document](https://evorbf.readthedocs.io/)
* [Download releases](https://pypi.org/project/evorbf/) 
* [Issue tracker](https://github.com/thieu1995/evorbf/issues) 
* [Notable changes log](/ChangeLog.md)
* [Official discussion group](https://t.me/+fRVCJGuGJg1mNDg1)
