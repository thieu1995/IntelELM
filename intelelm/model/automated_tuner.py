#!/usr/bin/env python
# Created by "Thieu" at 18:32, 16/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from intelelm import MhaElmRegressor, MhaElmClassifier


class AutomatedMhaElmTuner:

    def __init__(self, task="classification", param_dict=None, search_method="gridsearch", **kwargs):
        """
        Initializes the tuner

        Args:
            task (str): The task to be tuned (e.g., classification or regression).
            param_dict (dict): The parameter grid or distributions for hyperparameter tuning.
            optimization_method (str): The method for tuning (e.g., 'gridsearch', 'randomsearch').
            **kwargs: Additional arguments for tuning methods like cv, n_iter, etc.
        """
        if task == "classification":
            self.model_class = MhaElmClassifier
        else:
            self.model_class = MhaElmRegressor
        self.param_dict = param_dict
        self.search_method = search_method
        self.kwargs = kwargs
        self.best_estimator_ = None
        self.best_params_ = None

    def _grid_search(self, X, y):
        """
        Performs GridSearchCV to tune hyperparameters.

        Args:
            X (array-like): Training features.
            y (array-like): Training target values.
        """
        grid_search = GridSearchCV(estimator=self.model_class(), param_grid=self.param_dict, **self.kwargs)
        grid_search.fit(X, y)
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

    def _random_search(self, X, y):
        """
        Performs RandomizedSearchCV to tune hyperparameters.

        Args:
            X (array-like): Training features.
            y (array-like): Training target values.
        """
        random_search = RandomizedSearchCV(estimator=self.model_class(), param_distributions=self.param_dict, **self.kwargs)
        random_search.fit(X, y)
        self.best_estimator_ = random_search.best_estimator_
        self.best_params_ = random_search.best_params_

    def fit(self, X, y):
        """
        Fits the tuner to the data and tunes the hyperparameters.

        Args:
            X (array-like): Training features.
            y (array-like): Training target values.

        Returns:
            self: Fitted tuner object.
        """
        if self.search_method == "gridsearch":
            self._grid_search(X, y)
        elif self.search_method == "randomsearch":
            self._random_search(X, y)
        else:
            raise ValueError(f"Unsupported optimization method: {self.search_method}")

        return self

    def predict(self, X):
        """
        Predicts using the best estimator found during tuning.

        Args:
            X (array-like): Data to predict.

        Returns:
            array-like: Predicted values.
        """
        if self.best_estimator_ is None:
            raise NotFittedError("This AutomatedMhaElmTuner instance is not fitted yet. Call 'fit' before using this estimator.")

        return self.best_estimator_.predict(X)
