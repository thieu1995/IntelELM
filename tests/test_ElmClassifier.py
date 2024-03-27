#!/usr/bin/env python
# Created by "Thieu" at 11:27, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import ElmClassifier


def test_ElmClassifier_class():
    X = np.random.rand(100, 6)
    y = np.random.randint(0, 2, size=100)

    model = ElmClassifier(hidden_size=10, act_name="elu", seed=42)
    model.fit(X, y)
    pred = model.predict(X)
    assert ElmClassifier.SUPPORTED_CLS_METRICS == model.SUPPORTED_CLS_METRICS
    assert pred[0] in (0, 1)
