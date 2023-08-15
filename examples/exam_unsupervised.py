#!/usr/bin/env python
# Created by "Thieu" at 22:28, 14/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from intelelm import get_dataset, MhaElmRegressor

data = get_dataset("gauss-50-12")
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)


opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
model = MhaElmRegressor(hidden_size=10, act_name="elu", obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras)
model.fit(data.X_train, data.y_train)

pred = model.predict(data.X_test)
print(pred)
