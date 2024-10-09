#!/usr/bin/env python
# Created by "Thieu" at 19:27, 16/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import get_dataset, AutomatedMhaElmTuner


data = get_dataset("diabetes")
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)

data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=('minmax', ))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=('minmax', ))
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))

# Example parameter grid
param_dict = {
    'layer_sizes': [10, 20],
    'act_name': ['relu', 'elu'],
    "obj_name": ["RMSE", "MAE"],
    'optim': ['BaseGA', "OriginalPSO"],
    'optim_paras__epoch': [10, 20],
    'optim_paras__pop_size': [20, 30],
    'seed': [42],
    "verbose": [False],
}

# Initialize the tuner
tuner = AutomatedMhaElmTuner(
    task="regression",
    param_dict=param_dict,
    search_method="randomsearch",  # or "randomsearch"
    scoring='r2',
    cv=3,
    verbose=2,          # Example additional argument
    random_state=42,    # Additional parameter for RandomizedSearchCV
    n_jobs=4            # Parallelization
)

# Perform tuning
tuner.fit(data.X_train, data.y_train)

print("Best Parameters: ", tuner.best_params_)
print("Best Estimator: ", tuner.best_estimator_)

pred = tuner.predict(data.X_test)
# print(pred)

print(tuner.best_estimator_.score(data.X_test, data.y_test, method="MSE"))
print(tuner.best_estimator_.score(data.X_test, data.y_test, method="MAPE"))
print(tuner.best_estimator_.score(data.X_test, data.y_test, method="R2"))
print(tuner.best_estimator_.scores(data.X_test, data.y_test, list_methods=["MSE", "MAPE", "R2", "KGE", "NSE"]))
