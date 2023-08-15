#!/usr/bin/env python
# Created by "Thieu" at 15:45, 15/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import MhaElmRegressor

np.random.seed(42)


def test_MhaElmRegressor_class():
    X = np.random.uniform(low=0.0, high=1.0, size=(100, 5))
    noise = np.random.normal(loc=0.0, scale=0.1, size=(100, 5))
    y = 2 * X + 1 + noise

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = MhaElmRegressor(hidden_size=10, act_name="elu", obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=False)
    model.fit(X, y)

    pred = model.predict(X)
    assert MhaElmRegressor.SUPPORTED_REG_OBJECTIVES == model.SUPPORTED_REG_OBJECTIVES
    assert len(pred) == X.shape[0]
