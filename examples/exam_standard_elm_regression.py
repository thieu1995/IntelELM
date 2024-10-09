#!/usr/bin/env python
# Created by "Thieu" at 10:12, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import get_dataset, ElmRegressor


data = get_dataset("diabetes")
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)

data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=('minmax', ))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=('minmax', ))
data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))

model = ElmRegressor(layer_sizes=(10, ), act_name="elu", seed=42)
model.fit(data.X_train, data.y_train)

y_pred = model.predict(data.X_test)

print(model.score(data.X_test, data.y_test, method="RMSE"))
print(model.scores(data.X_test, data.y_test, list_methods=("RMSE", "MAPE")))
print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=("MAPE", "R2", "NSE")))
