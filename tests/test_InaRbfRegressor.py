#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np

from evorbf import InaRbfRegressor

np.random.seed(42)


def test_InaRbfRegressor_class():
    X = np.random.uniform(low=0.0, high=1.0, size=(100, 5))
    noise = np.random.normal(loc=0.0, scale=0.1, size=(100, 5))
    y = 2 * X + 1 + noise

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = InaRbfRegressor(size_hidden=10, center_finder="kmean", regularization=False, lamda=0.01,
                            obj_name="MSE", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=True, seed=42, obj_weights=None)
    model.fit(X, y)

    pred = model.predict(X)
    assert InaRbfRegressor.SUPPORTED_REG_OBJECTIVES == model.SUPPORTED_REG_OBJECTIVES
    assert len(pred) == X.shape[0]
