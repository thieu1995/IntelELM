#!/usr/bin/env python
# Created by "Thieu" at 16:31, 16/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(21)


def make_data(n_samples=1000):
    x = np.array([i / 100 for i in range(n_samples)])
    r = [a/10 for a in x]
    y = np.sin(x) + np.random.uniform(-0.5, 0.2, len(x)) + np.array(r)
    return x, y

X, y = make_data(1000)
X = np.reshape(X, (-1, 1))

# plt.scatter(X, y, s=5, color="blue")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=11)


model = SVR().fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print("R-squared:", score)
print("MSE:", mean_squared_error(y_test, y_pred))

###########################################################################################################

import numpy as np
from intelelm import Data, MhaElmRegressor

data = Data().set_train_test(X_train, y_train, X_test, y_test)
opt_paras = {"name": "GA", "epoch": 100, "pop_size": 30}
model = MhaElmRegressor(hidden_size=10, act_name="elu", obj_name="MSE", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=False)
model.fit(data.X_train, data.y_train)

pred = model.predict(data.X_test)
print(pred)

print(model.score(data.X_test, data.y_test, method="MSE"))
print(model.score(data.X_test, data.y_test, method="MAPE"))
print(model.score(data.X_test, data.y_test, method="R2"))
print(model.score(data.X_test, data.y_test, method="NSE"))

print(model.scores(data.X_test, data.y_test, list_methods=["MSE", "MAPE", "R2", "KGE", "NSE"]))
