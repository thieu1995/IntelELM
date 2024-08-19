#!/usr/bin/env python
# Created by "Thieu" at 19:59, 19/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import get_dataset, AutomatedMhaElmComparator


data = get_dataset("diabetes")
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)

data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=('minmax', ))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=('minmax', ))
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))


# Example optimizer dict
optimizer_dict = {
    'BaseGA':       {"epoch": 10, "pop_size": 20},
    "OriginalPSO":  {"epoch": 10, "pop_size": 20},
}

# Initialize the comparator
compartor = AutomatedMhaElmComparator(
    optimizer_dict=optimizer_dict,
    task="regression",
    hidden_size=10,
    act_name="elu",
    obj_name="R2",
    verbose=False,
    seed=42,
)

# Perform comparison
# results = compartor.compare_cross_val_score(data.X_train, data.y_train, metric="RMSE", cv=4, n_trials=2, to_csv=True)
# print(results)

# results = compartor.compare_cross_validate(data.X_train, data.y_train, metrics=["MSE", "MAPE", "R2", "KGE", "NSE"],
#                                            cv=4, return_train_score=True, n_trials=2, to_csv=True)
# print(results)

results = compartor.compare_train_test(data.X_train, data.y_train, data.X_test, data.y_test, metrics=["MSE", "MAPE", "R2", "KGE", "NSE"],
                                       n_trials=2, to_csv=True)
print(results)
