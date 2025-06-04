#!/usr/bin/env python
# Created by "Thieu" at 22:28, 14/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import get_dataset, MhaElmRegressor

# Load and split the dataset
data = get_dataset("diabetes")
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)


# Scale the data
def scale_data(train_data, test_data, scaling_methods=('minmax',)):
    scaled_train, scaler = data.scale(train_data, scaling_methods=scaling_methods)
    scaled_test = scaler.transform(test_data)
    return scaled_train, scaled_test, scaler


data.X_train, data.X_test, scaler_X = scale_data(data.X_train, data.X_test)
data.y_train, data.y_test, scaler_y = scale_data(data.y_train, np.reshape(data.y_test, (-1, 1)))

# Define model parameters and initialize the model
opt_paras = {"name": "GA", "epoch": 100, "pop_size": 30}
model = MhaElmRegressor(layer_sizes=(10,), act_name="elu", obj_name="MSE",
                        optim="BaseGA", optim_params=opt_paras, seed=42,
                        lb=None, ub=None, mode='single', n_workers=None, termination=None)

# Train the model
model.fit(data.X_train, data.y_train)

# Make predictions
y_pred = model.predict(data.X_test)

# Evaluate the model
print(model.score(data.X_test, data.y_test, method="RMSE"))
print(model.scores(data.X_test, data.y_test, list_methods=("RMSE", "R2")))
print(model.evaluate(data.y_test, y_pred, list_metrics=("R2", "MAPE", "NSE")))
