#!/usr/bin/env python
# Created by "Thieu" at 22:32, 16/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from permetrics import ClassificationMetric
from intelelm import MhaElmClassifier

# Constants
TEST_SIZE = 0.25
RANDOM_STATE = 0
SVM_METRICS = ["AS", "RS", "PS", "F1S"]
MHAELM_METRICS = ["AS", "PS", "F1S", "CEL", "BSL"]
MHAELM_LAYER_SIZES = (10,)
MHAELM_ACT_NAME = "elu"
MHAELM_OBJ_NAME = "BSL"
MHAELM_OPTIM = "BaseGA"
GA_PARAMS = {"name": "GA", "epoch": 100, "pop_size": 30}
SEED = 42


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def evaluate_model(model, X_train, y_train, X_test, y_test, metrics):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = ClassificationMetric(y_test, y_pred)
    print(f"Results of {model.__class__.__name__} on Breast Cancer dataset!")
    print(cm.get_metrics_by_list_names(metrics))
    return model


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_train, X_test = standardize_data(X_train, X_test)

# Linear Support Vector Classifier
svc_model = LinearSVC()
evaluate_model(svc_model, X_train, y_train, X_test, y_test, SVM_METRICS)

# MhaElmClassifier
mhaelm_model = MhaElmClassifier(
    layer_sizes=MHAELM_LAYER_SIZES,
    act_name=MHAELM_ACT_NAME,
    obj_name=MHAELM_OBJ_NAME,
    optim=MHAELM_OPTIM,
    optim_params=GA_PARAMS,
    verbose=False,
    seed=SEED,
    lb=None, ub=None, mode='single', n_workers=None, termination=None
)
evaluate_model(mhaelm_model, X_train, y_train, X_test, y_test, SVM_METRICS)
print("Try my AS metric with score function")
print(mhaelm_model.score(X_test, y_test, method="AS"))
print("Try my multiple metrics with scores function")
print(mhaelm_model.scores(X_test, y_test, list_methods=MHAELM_METRICS))
