#!/usr/bin/env python
# Created by "Thieu" at 12:00, 16/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from permetrics import ClassificationMetric
from intelelm import MhaElmClassifier

#### Load dataset
cancer = load_breast_cancer()
# The data set is presented in a dictionary form:
print(cancer.keys())

df_feat = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
# cancer column is our target
df_target = pd.DataFrame(cancer['target'], columns =['Cancer'])

print("Feature Variables: ")
print(df_feat.info())

print("Dataframe looks like : ")
print(df_feat.head())


## Train Test Split
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size = 0.30, random_state = 101)

## Train the MhaElmClassifier without Hyper-parameter Tuning –
# train the model on train set
opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)
model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="BSL", optimizer="BaseGA", optimizer_paras=opt_paras, verbose=False)
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = ClassificationMetric(y_test, pred)
print(cm.get_metrics_by_list_names(["AS", "PS", "RS", "F1S"]))

print("================================================================================================")
######################################################################################################

from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'hidden_size': list(range(5, 50, 5)),
			'act_name': ["relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "swish",
			             "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink"],
			'obj_name': ['BSL', "HS"],
			'optimizer_paras': [
				{"name": "GA", "epoch": 10, "pop_size": 30},
				{"name": "GA", "epoch": 20, "pop_size": 30},
				{"name": "GA", "epoch": 30, "pop_size": 30},
				{"name": "GA", "epoch": 40, "pop_size": 30}
			]
}

grid = GridSearchCV(MhaElmClassifier(verbose=False), param_grid=param_grid, refit=True, cv=4, verbose=3)
# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
cm = ClassificationMetric(y_test, grid_predictions)
print(cm.get_metrics_by_list_names(["AS", "PS", "RS", "F1S"]))
