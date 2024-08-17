#!/usr/bin/env python
# Created by "Thieu" at 06:36, 17/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import get_dataset, AutomatedMhaElmTuner


data = get_dataset("circles")
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)

data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=('minmax', ))
data.X_test = scaler_X.transform(data.X_test)

# Example parameter grid
param_dict = {
    'hidden_size': [10, 20],
    'act_name': ['relu', 'elu'],
    "obj_name": ["BSL", "KLDL", "F1S"],
    'optimizer': ['BaseGA', "OriginalPSO"],
    'optimizer_paras__epoch': [10, 20],
    'optimizer_paras__pop_size': [20],
    'seed': [42],
    "verbose": [False],
}

# Initialize the tuner
tuner = AutomatedMhaElmTuner(
    task="classification",
    param_dict=param_dict,
    search_method="randomsearch",  # or "randomsearch"
    cv=3,  # Example additional argument
    scoring='accuracy',  # Example additional argument
    verbose=2
)

# Perform tuning
tuner.fit(data.X_train, data.y_train)

print("Best Parameters: ", tuner.best_params_)
print("Best Estimator: ", tuner.best_estimator_)

pred = tuner.predict(data.X_test)
# print(pred)

print(tuner.best_estimator_.score(data.X_test, data.y_test, method="AS"))
print(tuner.best_estimator_.score(data.X_test, data.y_test, method="PS"))
print(tuner.best_estimator_.score(data.X_test, data.y_test, method="RS"))
print(tuner.best_estimator_.scores(data.X_test, data.y_test, list_methods=["AS", "PS", "RS", "F1S", "NPV"]))
