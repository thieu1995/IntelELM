#!/usr/bin/env python
# Created by "Thieu" at 16:31, 16/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from intelelm import Data, MhaElmRegressor

# Constants
N_SAMPLES = 1000
TEST_SIZE = 0.2
RANDOM_STATE = 11


def make_data(n_samples=N_SAMPLES):
    x = np.array([i / 100 for i in range(n_samples)])
    r = [a / 10 for a in x]
    y = np.sin(x) + np.random.uniform(-0.5, 0.2, len(x)) + np.array(r)
    return x, y


def train_and_evaluate_model(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X = np.reshape(X, (-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True,
                                                        random_state=random_state)
    model = SVR().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)

    print(f"SVR results: R2 = {score:.3f}, MSE = {mean_squared_error(y_test, y_pred):.3f}")


def train_and_evaluate_mha_elm(X_train, y_train, X_test, y_test):
    data = Data().set_train_test(X_train, y_train, X_test, y_test)
    opt_paras = {"name": "GA", "epoch": 100, "pop_size": 30}
    model = MhaElmRegressor(layer_sizes=(10,), act_name="elu", obj_name="MSE", optim="BaseGA",
                            optim_params=opt_paras, verbose=False, seed=42,
                            lb=None, ub=None, mode='single', n_workers=None, termination=None)
    model.fit(data.X_train, data.y_train)
    pred = model.predict(data.X_test)

    print(pred)
    print(model.score(data.X_test, data.y_test, method="MSE"))
    print(model.score(data.X_test, data.y_test, method="MAPE"))
    print(model.score(data.X_test, data.y_test, method="R2"))
    print(model.score(data.X_test, data.y_test, method="NSE"))
    print(model.scores(data.X_test, data.y_test, list_methods=["MSE", "MAPE", "R2", "KGE", "NSE"]))


# Generating Data
X, y = make_data()
train_and_evaluate_model(X, y)

# Split data for MhaElmRegressor
X = np.reshape(X, (-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE)
train_and_evaluate_mha_elm(X_train, y_train, X_test, y_test)
