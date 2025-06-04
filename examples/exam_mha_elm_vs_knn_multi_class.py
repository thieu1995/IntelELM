#!/usr/bin/env python
# Created by "Thieu" at 07:37, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

## Kaggle Dataset: https://www.kaggle.com/code/gideon94/multiclass-classification-svm-knn-dt-comparison/notebook
## Multiclass classification based on hazelnuts variety "c_avellana, c_americana,
## c_corutana and comparting with SVM, kNN, Decision tree, Naive Bayes classifiers

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from permetrics import ClassificationMetric
from intelelm import MhaElmClassifier

DATA_FILE = "hazelnuts.txt"
KNN_NEIGHBORS = 25
TEST_SIZE = 0.25
RANDOM_STATE = 0
OPTIMIZATION_PARAMETERS = {"name": "GA", "epoch": 100, "pop_size": 30}


def load_and_preprocess_data(filename):
    hazel_df = pd.read_csv(filename, sep="\t", header=None).transpose()
    hazel_df.columns = ["sample_id", "length", "width", "thickness", "surface_area", "mass", "compactness",
                        "hardness", "shell_top_radius", "water_content", "carbohydrate_content", "variety"]
    features = hazel_df.drop(["variety", "sample_id"], axis=1)
    target = hazel_df["variety"]
    X_train, X_test, y_train, y_test = train_test_split(features.values.astype(float), target.values,
                                                        test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test, y_train, y_test):
    scaler_X = MinMaxScaler()
    scaler_y = LabelEncoder()
    X_train_scaled, X_test_scaled = scaler_X.fit_transform(X_train), scaler_X.transform(X_test)
    y_train_encoded, y_test_encoded = scaler_y.fit_transform(y_train), scaler_y.transform(y_test)
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded


def evaluate_model(model, X_test, y_test, description):
    y_pred = model.predict(X_test)
    cm = ClassificationMetric(y_test, y_pred)
    print(f"Results of {description}!")
    print(cm.get_metrics_by_list_names(["AS", "RS", "PS", "F1S"]))
    if description == "MhaElmClassifier":
        print("Try my AS metric with score function")
        print(model.score(X_test, y_test, method="AS"))
        print("Try my multiple metrics with scores function")
        print(model.scores(X_test, y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))


X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_FILE)
X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = scale_data(X_train, X_test, y_train, y_test)

print(np.unique(y_train_encoded))
print(len(np.unique(y_test_encoded)))

knn_model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
knn_model.fit(X_train_scaled, y_train_encoded)
evaluate_model(knn_model, X_test_scaled, y_test_encoded, "KNN")

elm_model = MhaElmClassifier(layer_sizes=(10,), act_name="elu", obj_name="BSL", optim="BaseGA",
                             optim_params=OPTIMIZATION_PARAMETERS, verbose=False, seed=42)
elm_model.fit(X_train_scaled, y_train_encoded)
evaluate_model(elm_model, X_test_scaled, y_test_encoded, "MhaElmClassifier")
