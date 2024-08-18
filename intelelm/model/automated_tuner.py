#!/usr/bin/env python
# Created by "Thieu" at 18:32, 16/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from intelelm import MhaElmRegressor, MhaElmClassifier
from intelelm.utils.evaluator import get_metric_sklearn


class AutomatedMhaElmTuner:
    """
    Automated hyperparameter tuner for MhaElm models.

    Performs hyperparameter tuning for MhaElm models using either GridSearchCV or RandomizedSearchCV.
    Provides an interface for fitting and predicting using the best found model.

    Attributes:
        model_class (class): The MhaElm model class (MhaElmRegressor or MhaElmClassifier).
        param_grid (dict): The parameter grid for hyperparameter tuning.
        search_method (str): The optimization method ('gridsearch' or 'randomsearch').
        kwargs (dict): Additional keyword arguments for the search method.
        searcher (GridSearchCV or RandomizedSearchCV): The searcher
        best_estimator_ (sklearn.base.BaseEstimator): The best estimator found during tuning.
        best_params_ (dict): The best hyperparameters found during tuning.

    Methods:
        fit(X, y): Fits the tuner to the data and tunes hyperparameters.
        predict(X): Predicts using the best estimator.
    """

    def __init__(self, task="classification", param_dict=None, search_method="gridsearch", scoring=None, cv=3, **kwargs):
        """
        Initializes the tuner

        Args:
            task (str): The task to be tuned (e.g., classification or regression).
            param_dict (dict): The parameter grid or distributions for hyperparameter tuning.
            optimization_method (str): The method for tuning (e.g., 'gridsearch', 'randomsearch').
            **kwargs: Additional arguments for tuning methods like cv, n_iter, etc.
        """
        self.task = task
        if task not in ("classification", "regression"):
            raise ValueError(f"Invalid task type: {task}. Supported tasks are 'classification' and 'regression'.")
        self.model_class = MhaElmClassifier if task == "classification" else MhaElmRegressor
        self.param_dict = param_dict
        self.search_method = search_method.lower()
        self.scoring = scoring
        self.cv = cv
        self.kwargs = kwargs
        self.searcher = None
        self.best_estimator_ = None
        self.best_params_ = None

    def _get_search_object(self):
        """
        Returns an instance of GridSearchCV or RandomizedSearchCV based on the chosen search method.

        Raises:
            ValueError: If an unsupported search method is specified or if a parameter grid is missing for RandomizedSearchCV.
        """
        if not self.param_dict:
            raise ValueError("Searching hyper-parameter requires a param_dict as a dictionary.")
        if not self.scoring:
            raise ValueError("Searching hyper-parameter requires a scoring method.")
        metrics = get_metric_sklearn(task=self.task, metric_names=[self.scoring])
        if len(metrics) == 1:
            self.scoring = metrics[self.scoring]
        if self.search_method == "gridsearch":
            return GridSearchCV(estimator=self.model_class(), param_grid=self.param_dict,
                                scoring=self.scoring, cv=self.cv, **self.kwargs)
        elif self.search_method == "randomsearch":
            return RandomizedSearchCV(estimator=self.model_class(), param_distributions=self.param_dict,
                                      scoring=self.scoring, cv=self.cv, **self.kwargs)
        else:
            raise ValueError(f"Unsupported searching method: {self.search_method}")

    def fit(self, X, y):
        """
        Fits the tuner to the data and tunes the hyperparameters.

        Args:
            X (array-like): Training features.
            y (array-like): Training target values.

        Returns:
            self: Fitted tuner object.
        """
        self.searcher = self._get_search_object()
        self.searcher.fit(X, y)
        self.best_estimator_ = self.searcher.best_estimator_
        self.best_params_ = self.searcher.best_params_
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
