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


#Loading data
hazel_df = pd.read_csv("hazelnuts.txt", sep="\t", header=None)
hazel_df = hazel_df.transpose()
hazel_df.columns = ["sample_id", "length", "width", "thickness", "surface_area", "mass", "compactness",
                    "hardness", "shell_top_radius", "water_content", "carbohydrate_content", "variety"]
print(hazel_df.head())

all_features = hazel_df.drop(["variety","sample_id"],axis=1)
target_feature = hazel_df["variety"]
print(all_features.head())

#Dataset preprocessing
X = all_features.values.astype(float) # returns a numpy array of type float
y = target_feature.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

scaler_X = MinMaxScaler()
scaler_y = LabelEncoder()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

print(np.unique(y_train))
print(len(np.unique(y_test)))

##### KNN
model = KNeighborsClassifier(n_neighbors = 25)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = ClassificationMetric(y_test, y_pred, decimal=6)
print("Results of KNN!")
print(cm.get_metrics_by_list_names(["AS", "RS", "PS", "F1S"]))

# ###################################################################################################

opt_paras = {"name": "GA", "epoch": 100, "pop_size": 30}
model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="BSL", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = ClassificationMetric(y_test, y_pred, decimal=6)
print("Results of MhaElmClassifier!")
print(cm.get_metrics_by_list_names(["AS", "RS", "PS", "F1S"]))

print("Try my AS metric with score function")
print(model.score(X_test, y_test, method="AS"))

print("Try my multiple metrics with scores function")
print(model.scores(X_test, y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))
