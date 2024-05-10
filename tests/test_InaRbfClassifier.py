#!/usr/bin/env python
# Created by "Thieu" at 17:13, 10/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np

from evorbf import InaRbfClassifier


def test_InaRbfClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = InaRbfClassifier(size_hidden=25, center_finder="kmean", regularization=False, lamda=0.5, obj_name="NPV",
                             optimizer="OriginalWOA", optimizer_paras=opt_paras, verbose=True, seed=42)
    model.fit(X, y)
    pred = model.predict(X)
    assert InaRbfClassifier.SUPPORTED_CLS_OBJECTIVES == model.SUPPORTED_CLS_OBJECTIVES
    assert pred[0] in (0, 1)
