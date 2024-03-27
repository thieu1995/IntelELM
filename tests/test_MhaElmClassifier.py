#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import MhaElmClassifier


def test_MhaElmClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="AS", optimizer="BaseGA",
                             optimizer_paras=opt_paras, verbose=False, seed=42)
    model.fit(X, y)
    pred = model.predict(X)
    assert MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES == model.SUPPORTED_CLS_OBJECTIVES
    assert pred[0] in (0, 1)
