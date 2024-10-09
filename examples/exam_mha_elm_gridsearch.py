#!/usr/bin/env python
# Created by "Thieu" at 12:00, 16/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from intelelm import MhaElmClassifier

# Constants
TEST_SIZE = 0.30
RANDOM_STATE = 101
OPT_PARAMS = {"name": "GA", "epoch": 10, "pop_size": 30}
LAYER_SIZES = (10,)
ACTIVATION_FUNCTION = "elu"
OBJECTIVE_NAME = "BSL"
OPTIMIZER = "BaseGA"
SEED = 42
VERBOSE = False


def load_and_preprocess_data():
	"""Loads and preprocesses the breast cancer dataset into features and target."""
	cancer_data = load_breast_cancer()
	features = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
	target = pd.DataFrame(cancer_data['target'], columns=['Cancer'])
	return features, target


def train_test_split_data(features, target):
	"""Splits the dataset into training and testing sets."""
	return train_test_split(features, np.ravel(target), test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_model(X_train, y_train):
	"""Trains the MhaElmClassifier model."""
	print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)
	model = MhaElmClassifier(layer_sizes=LAYER_SIZES, act_name=ACTIVATION_FUNCTION, obj_name=OBJECTIVE_NAME,
							 optim=OPTIMIZER, optim_paras=OPT_PARAMS, verbose=VERBOSE, seed=SEED)
	model.fit(X_train, y_train)
	return model


def evaluate_model(model, X_test, y_test):
	"""Evaluates the model on the test set."""
	predictions = model.predict(X_test)
	metrics = model.evaluate(y_test, predictions, list_metrics=["AS", "PS", "RS", "F1S"])
	print(metrics)


def perform_grid_search(X_train, y_train, X_test, y_test):
	"""Performs grid search for hyper-parameter tuning."""
	param_grid = {
		'layer_sizes': [tuple([size]) for size in range(5, 50, 5)],
		'act_name': ["relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
					 "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink"],
		'obj_name': ['BSL', "HS"],
		'optim_paras': [
			{"name": "GA", "epoch": 10, "pop_size": 30},
			{"name": "GA", "epoch": 20, "pop_size": 30},
			{"name": "GA", "epoch": 30, "pop_size": 30},
			{"name": "GA", "epoch": 40, "pop_size": 30}
		]
	}
	grid_search = GridSearchCV(MhaElmClassifier(verbose=VERBOSE), param_grid=param_grid, refit=True, cv=4, verbose=3)
	grid_search.fit(X_train, y_train)
	print(grid_search.best_params_)
	print(grid_search.best_estimator_)
	grid_predictions = grid_search.predict(X_test)
	print(model.evaluate(y_test, grid_predictions, list_metrics=["AS", "PS", "RS", "F1S"]))


# Main execution flow
features, target = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split_data(features, target)
model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
perform_grid_search(X_train, y_train, X_test, y_test)
