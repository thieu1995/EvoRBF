============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/evorbf />`_::

   $ pip install evorbf==1.0.1


* Install directly from source code::

   $ git clone https://github.com/thieu1995/evorbf.git
   $ cd evorbf
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/evorbf


After installation, you can import EvoRBF as any other Python module::

   $ python
   >>> import evorbf
   >>> evorbf.__version__

========
Examples
========

In this section, we will explore the usage of the EvoRBF model with the assistance of a dataset. While all the
preprocessing steps mentioned below can be replicated using Scikit-Learn, we have implemented some utility functions
to provide users with convenience and faster usage::

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



A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize
the data within a particular range.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
