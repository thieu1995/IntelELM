#!/usr/bin/env python
# Created by "Thieu" at 11:15, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from permetrics import ClassificationMetric
from intelelm import ElmClassifier


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metric = ClassificationMetric(y_test, y_pred)
    results = metric.get_metrics_by_list_names(["AS", "RS", "PS", "F1S"])
    return results, model, y_pred


# Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train, X_test = scale_data(X_train, X_test)

# Train and evaluate LinearSVC model
linear_svc = LinearSVC()
linear_svc_results, linear_svc_model, linear_svc_preds = train_and_evaluate_model(linear_svc, X_train, X_test, y_train, y_test)
print("Results of LinearSVC on Breast Cancer dataset!")
print(linear_svc_results)

# Train and evaluate ElmClassifier model
elm_classifier = ElmClassifier(layer_sizes=(10,), act_name="elu", seed=42)
elm_classifier_results, elm_classifier_model, elm_classifier_preds = train_and_evaluate_model(elm_classifier, X_train,
                                                                                              X_test, y_train, y_test)
print("Results of ElmClassifier on Breast Cancer dataset!")
print(elm_classifier_results)
print("Try my AS metric with score function:", elm_classifier_model.score(X_test, y_test, method="AS"))
print("Try my multiple metrics with scores function:",
      elm_classifier_model.scores(X_test, y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))

