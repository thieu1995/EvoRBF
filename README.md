
<p align="center">
<img style="max-width:100%;" src="https://thieu1995.github.io/post/2023-08/evorbf1.png" alt="EvoRBF"/>
</p>

---

[![GitHub release](https://img.shields.io/badge/release-2.0.0-yellow.svg)](https://github.com/thieu1995/evorbf/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/evorbf) 
[![PyPI version](https://badge.fury.io/py/evorbf.svg)](https://badge.fury.io/py/evorbf)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/evorbf.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/evorbf.svg)
[![Downloads](https://static.pepy.tech/badge/evorbf)](https://pepy.tech/project/evorbf)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/evorbf/actions/workflows/publish-package.yml/badge.svg)](https://github.com/thieu1995/evorbf/actions/workflows/publish-package.yaml)
[![Documentation Status](https://readthedocs.org/projects/evorbf/badge/?version=latest)](https://evorbf.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11136007.svg)](https://doi.org/10.5281/zenodo.11136007)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## 🌟 Introduction

**EvoRBF** is a powerful Python library for training Radial Basis Function (RBF) networks using nature-inspired algorithms (NIAs). 
It provides a wide range of RBF models from traditional to advanced, and supports hyperparameter tuning using optimizers 
like Whale Optimization Algorithm (WOA), Genetic Algorithm (GA), and more.

---

## 🚀 Key Features

| **EvoRBF**                           | **Evolving Radial Basis Function Network**             |
|--------------------------------------|--------------------------------------------------------|
| **Free software**                    | GNU General Public License (GPL) V3 license            |
| **Traditional RBF models**           | `RbfRegressor`, `RbfClassifier`                        |
| **Advanced RBF models**              | `AdvancedRbfRegressor`, `AdvancedRbfClassifier`        | 
| **Nature-inspired RBF models**       | `NiaRbfRegressor`, `NiaRbfClassifier`                  |
| **Tuner for traditional RBF models** | `NiaRbfTuner`                                          | 
| **Provided total ML models**         | \> 400 Models                                          |
| **Supported total metrics**          | \>= 67 (47 regressions and 20 classifications)         |
| **Supported loss functions**         | \>= 61 (45 regressions and 16 classifications)         |
| **Documentation**                    | https://evorbf.readthedocs.io                          | 
| **Python versions**                  | \>= 3.8.x                                              |  
| **Dependencies**                     | numpy, scipy, scikit-learn, pandas, mealpy, permetrics |


## 🧾 Citation

Please include these citations if you plan to use this library:

```bibtex
@software{thieu_2024_11136008,
  author       = {Nguyen Van Thieu},
  title        = {EvoRBF: A Nature-inspired Algorithmic Framework for Evolving Radial Basis Function Networks},
  month        = June,
  year         = 2025,
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

## 📦 Installation

Install the latest version from PyPI:

```bash
pip install evorbf
```

Verify installation:

```bash
$ python
>>> import evorbf
>>> evorbf.__version__
```


We have provided above several ways to import and call the proposed classes. If you need more details how to 
use each of them, please check out the folder [examples](/examples). In this short demonstration, we will use 
Whale Optimization Algorithm to optimize the `sigmas` (in non-linear Gaussian kernel) and `reg_lambda` of 
L2 regularization in RBF network (WOA-RBF model) for Diabetes prediction problem.

```python
import numpy as np
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
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", ))
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))

## Create model
opt_paras = {"name": "WOA", "epoch": 50, "pop_size": 20}
model = NiaRbfRegressor(size_hidden=25,             # Set up big enough hidden size 
                        center_finder="kmeans",     # Use KMeans to find the centers
                        regularization=True,        # Use L2 regularization 
                        obj_name="MSE",             # Mean squared error as fitness function for NIAs
                        optim="OriginalWOA",        # Use Whale Optimization
                        optim_params={"epoch": 50, "pop_size": 20},  # Set up parameter for Whale Optimization
                        verbose=True, seed=42)

## Train the model
model.fit(data.X_train, data.y_train)

## Test the model
y_pred = model.predict(data.X_test)

print(model.optimizer.g_best.solution)
## Calculate some metrics
print(model.score(X=data.X_test, y=data.y_test))
print(model.scores(X=data.X_test, y=data.y_test, list_metrics=["R2", "R", "KGE", "MAPE"]))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=["MSE", "RMSE", "R2S", "NSE", "KGE", "MAPE"]))
```

## 📚 Brief Theory Combine With Deep Usage

**EvoRBF** is mind-blowing framework for Radial Basis Function (RBF) networks.
We explain several keys components and provide several types of RBF networks that you will never see in other places.

You can read several papers by using Google Scholar search. There are many ways we can use Nature-inspired Algorithms 
to optimize Radial Basis Function network, for example, you can read [this paper](https://doi.org/10.1016/B978-0-443-18764-3.00015-1).
Here we will walk through some basic concepts and parameters that matter to this network.

### Structure

The RBF network consists of three layers:

1. **Input layer**: Accepts input features.
2. **Hidden layer**: Applies radial basis functions (e.g., Gaussian).
3. **Output layer**: Computes linear combinations of hidden outputs.

### 🛠️ Model Training

#### Traditional RBF models

In case of traditional RBF model. There are a few parameters need to identify to get the best model.
```code
1. The number of hidden nodes in hidden layer
2. The centers and widths (sigmas) of Gaussian function
3. The output weights
4. The regularization factor (lambda) L2
```

To train their parameters, 
```code
1. Using hyper-parameter tuning model such as GridSearchCV or RandomizedSearchCV to get the best hidden nodes
2. The centers can be calculated by Random or KMeans or unsupervised learning algorithms
3. The widths (sigmas) can be computed by hyper-parameter tuning process.
   + Width can be a single value that represent all hidden nodes has the same curve of Gaussian function
   + Width can be multiple values that each hidden node has a different value.
4. The output weights can be calculated by Moore-Penrose inverse (Matrix multiplication). Do not use Gradient Descent.
5. When setting regularization L2. lambda can be computed by hyper-parameter tuning process.
```

Example,
```python
from evorbf import RbfRegressor, RbfClassifier

model = RbfClassifier(size_hidden=10, center_finder="kmeans", sigmas=2.0, reg_lambda=0.1, seed=None)
model = RbfRegressor(size_hidden=4, center_finder="random", sigmas=(1.5, 2, 2, 2.5), reg_lambda=0, seed=42)

model.fit(X=X_train, y=y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
```


#### Advanced RBF models

In case of advanced RBF model. User can have so many different options.
```code
1. Choice different RBF kernel function such as Multiquadric (MQ), Inverse Multiquadric (IMQ), Thin Plate Spline (TPS), Exponential, Power,...
2. Choice different unsupervised learning algorithms to calculate the centers, and may be the number of hidden nodes.
   + For example, KMeans, or random algorithms, you need to set up the number of hidden nodes.
   + But, for MeanShift or DBSCAN algorithms, you don't need to set that value. They can automatically identify the number of cluters (number of hidden nodes).
3. This version may have the bias in output layer. 
```

Examples,
```python
from evorbf import AdvancedRbfClassifier, AdvancedRbfRegressor

model = AdvancedRbfClassifier(center_finder="random", finder_params={"n_centers": 15},
                 rbf_kernel="gaussian", kernel_params={"sigma": 1.5},
                 reg_lambda=0.1, has_bias=True, seed=42)

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

model.fit(X=X_train, y=y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
```

#### Nature-inspired Algorithm-based RBF models

This is the main purpose of this library. In this type of models,

```code
1. We use Nature-inspired Algorithm (NIA) to train widths (sigmas) value for each hidden node.
2. If you set up the Regularization technique, then NIA is automatically calculated the lambda factor
```

Examples,
```python
from evorbf import NiaRbfRegressor, NiaRbfClassifier

model = NiaRbfClassifier(size_hidden=25, center_finder="kmeans", 
                         regularization=False, obj_name="F1S",
                         optim="OriginalWOA", 
                         optim_params={"epoch": 50, "pop_size": 20}, 
                         verbose=True, seed=42)

model = NiaRbfRegressor(size_hidden=10, center_finder="random", 
                         regularization=True, obj_name="AS",
                         optim="BaseGA", 
                         optim_params={"epoch": 50, "pop_size": 20}, 
                         verbose=True, seed=42)

model.fit(X=X_train, y=y_train)
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)
```

#### 🎯 Nature-inspired Algorithm-based hyperparameter RBF tuning model

In this case, user can use NIA to tune hyper-parameters of traditional RBF models.

```python
from evorbf import NiaRbfTuner, IntegerVar, StringVar, FloatVar

# Design the boundary (for hyper-parameters)
my_bounds = [
    IntegerVar(lb=5, ub=21, name="size_hidden"),
    StringVar(valid_sets=("kmeans", "random"), name="center_finder"),
    FloatVar(lb=(0.01,), ub=(3.0,), name="sigmas"),
    FloatVar(lb=(0, ), ub=(1.0, ), name="reg_lambda"),
]

model = NiaRbfTuner(problem_type="classification", bounds=my_bounds, cv=3, scoring="AS",
                    optim="OriginalWOA", optim_params={"epoch": 10, "pop_size": 20}, 
                    verbose=True, seed=42)
```

### My notes

1. RBF networks require training of both the centers and the widths of the Gaussian activation functions (this is the 1st phase of training).
2. RBF typically uses KMeans to find the centers:
   + This increases both complexity and computation time.
   + In this case, users need to define the widths: You can use a single global width or assign individual widths per hidden node.
   + Alternatively, centers can be chosen randomly, but this often fails to properly separate the data into meaningful clusters.
3. RBF also requires training the output weights (this is the 2nd phase).
4. Unlike MLP networks, RBF does not use gradient descent to compute output weights. Instead, it uses the Moore–Penrose pseudoinverse 
(via matrix multiplication and the least squares method) → This makes it faster than MLPs.
5. The Moore–Penrose inverse finds an exact solution, so there is no need for gradient descent or approximation algorithms in this step.
6. If overfitting occurs, you can apply L2 regularization to control the model's complexity.
7. For large-scale datasets, you should:
   + Increase the number of hidden nodes.
   + Increase the L2 regularization parameter to avoid overfitting.


```code
1. RbfRegressor, RbfClassifier:
   - You need to configure 4 types of hyperparameters.
   
2. AdvancedRbfRegressor, AdvancedRbfClassifier:
   - These models require setting 6 types of hyperparameters.
   - However, they offer much more flexibility—you can design your own custom RBF architectures.
   - For example:
     + RBF with bias in the output layer.
     + RBF using DBSCAN for center initialization and an exponential kernel function.

3. NiaRbfRegressor, NiaRbfClassifier:
   - You only need to set the hidden size.
   - These are the best-performing classes in this library.
     + Widths (sigmas) are automatically computed for each hidden node.
     + The regularization factor is automatically tuned to find the optimal value.

4. NiaRbfTuner:
   - Extremely useful for traditional RBF models.
   - It can automatically tune the hidden size.
   - However, only a single sigma value will be used across all hidden nodes.
```


## 📎 Official channels

* 🔗 [Official source code repository](https://github.com/thieu1995/evorbf)
* 📘 [Official document](https://evorbf.readthedocs.io/)
* 📦 [Download releases](https://pypi.org/project/evorbf/) 
* 🐞 [Issue tracker](https://github.com/thieu1995/evorbf/issues) 
* 📝 [Notable changes log](/ChangeLog.md)
* 💬 [Official discussion group](https://t.me/+fRVCJGuGJg1mNDg1)

---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=EvoRBF_QUESTIONS) @ 2025
