#!/usr/bin/env python
# Created by "Thieu" at 10:16 PM, 08/10/2024 --------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import OneHotEncoder
from intelelm.model.base_elm import BaseMhaElm, MultiLayerELM
from intelelm.utils.encoder import ObjectiveScaler


class MhaElmRegressor(BaseMhaElm, RegressorMixin):
    """
    MhaElmRegressor Class
    =====================
    A regression model based on the `BaseMhaElm` class, designed for optimization-based training
    using various optimizers and regression metrics. It supports flexible network creation and
    evaluation using the Permetrics library.

    Attributes
    ----------
    obj_weights : list, tuple, or np.ndarray, optional
        Weights for multiple objectives in multi-output regression.

    Methods
    -------
    __init__(layer_sizes=(10,), act_name="elu", obj_name="MSE", optim="BaseGA", optim_params=None, seed=None,
             verbose=False, obj_weights=None, lb=None, ub=None, mode='single', n_workers=None, termination=None)
        Initializes the `MhaElmRegressor` with specified parameters.

    create_network(X, y)
        Creates and initializes the regression network based on the input data and target values.

    objective_function(solution=None)
        Evaluates the fitness function for regression metrics based on the provided solution.

    score(X, y, method="RMSE")
        Returns the selected regression metric for the given test data and labels.

    scores(X, y, list_methods=("MSE", "MAE"))
        Returns a dictionary of multiple regression metrics for the given test data and labels.

    evaluate(y_true, y_pred, list_metrics=("MSE", "MAE"))
        Evaluates the performance of predictions using a list of regression metrics.
    """

    def __init__(self, layer_sizes=(10, ), act_name="elu", obj_name="MSE",
                 optim="BaseGA", optim_params=None, seed=None, verbose=False, obj_weights=None,
                 lb=None, ub=None, mode='single', n_workers=None, termination=None):
        super().__init__(layer_sizes=layer_sizes, act_name=act_name, obj_name=obj_name,
                         optim=optim, optim_params=optim_params, seed=seed, verbose=verbose,
                         lb=lb, ub=ub, mode=mode, n_workers=n_workers, termination=termination)
        self.obj_weights = obj_weights

    def create_network(self, X, y) -> MultiLayerELM:
        """
        Parameters
        ----------
        X : array-like
            The input data used to train the network.

        y : array-like
            The target values. It should be a 1D vector or a 2D matrix.
            Accepted types: list, tuple, np.ndarray.

        Returns
        -------
        MultiLayerELM
            An instance of MultiLayerELM initialized with the specified parameters and objective scaler.

        Raises
        ------
        TypeError
            If `y` is not a list, tuple, or np.ndarray, or if `y` has an invalid shape that is not a 1D vector or 2D matrix, or
            if `self.obj_weights` is of an invalid type.

        ValueError
            If the length of `self.obj_weights` does not match the number of objectives.
        """
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                size_output = 1
            elif y.ndim == 2:
                size_output = y.shape[1]
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector or 2D matrix.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        if size_output > 1:
            if self.obj_weights is None:
                self.obj_weights = 1./size_output * np.ones(size_output)
            elif self.obj_weights in (list, tuple, np.ndarray):
                if not (len(self.obj_weights) == size_output):
                    raise ValueError(f"There is {size_output} objectives, but obj_weights has size of {len(self.obj_weights)}")
            else:
                raise TypeError("Invalid obj_weights array type, it should be list, tuple or np.ndarray")
        network = MultiLayerELM(layer_sizes=self.layer_sizes, act_name=self.act_name, seed=self.seed)
        network.obj_scaler = ObjectiveScaler(obj_name="self", ohe_scaler=None)
        network.input_size = X.shape[1]
        return network

    def objective_function(self, solution=None):
        """
        Evaluates the fitness function for regression metric

        Parameters
        ----------
        solution : np.ndarray, default=None

        Returns
        -------
        result: float
            The fitness value
        """
        self.network.decode(solution, self.X_temp, self.y_temp)
        y_pred = self.network.predict(self.X_temp)
        loss_train = RegressionMetric(self.y_temp, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train

    def score(self, X, y, method="RMSE"):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseElm__score_reg(X, y, method)

    def scores(self, X, y, list_methods=("MSE", "MAE")):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseElm__scores_reg(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("MSE", "MAE")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseElm__evaluate_reg(y_true, y_pred, list_metrics)


class MhaElmClassifier(BaseMhaElm, ClassifierMixin):
    """
    MhaElmClassifier Class
    =======================
    A classification model based on the `BaseMhaElm` class, designed for optimization-based training
    using various optimizers and classification metrics. It supports flexible network creation and
    evaluation using the Permetrics library.

    Attributes
    ----------
    CLS_OBJ_LOSSES : list
        List of classification objective losses supported by the model.

    return_prob : bool
        Indicates whether the model should return probabilities for multi-class classification.

    Methods
    -------
    __init__(layer_sizes=(10,), act_name="elu", obj_name="F1S", optim="BaseGA", optim_params=None, seed=None,
             verbose=False, lb=None, ub=None, mode='single', n_workers=None, termination=None)
        Initializes the `MhaElmClassifier` with specified parameters.

    _check_y(y)
        Validates the target labels and determines the number of unique labels.

    create_network(X, y)
        Creates and initializes the classification network based on the input data and target labels.

    objective_function(solution=None)
        Evaluates the fitness function for classification metrics based on the provided solution.

    score(X, y, method="AS")
        Returns the selected classification metric for the given test data and labels.

    scores(X, y, list_methods=("AS", "RS"))
        Returns a dictionary of multiple classification metrics for the given test data and labels.

    evaluate(y_true, y_pred, list_metrics=("AS", "RS"))
        Evaluates the performance of predictions using a list of classification metrics.
    """

    CLS_OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, layer_sizes=(10, ), act_name="elu", obj_name="F1S",
                 optim="BaseGA", optim_params=None, seed=None, verbose=False,
                 lb=None, ub=None, mode='single', n_workers=None, termination=None):
        super().__init__(layer_sizes=layer_sizes, act_name=act_name, obj_name=obj_name,
                         optim=optim, optim_params=optim_params, seed=seed, verbose=verbose,
                         lb=lb, ub=ub, mode=mode, n_workers=n_workers, termination=termination)
        self.return_prob = False

    def _check_y(self, y):
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                return len(np.unique(y)), np.unique(y)
            raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")

    def create_network(self, X, y) -> MultiLayerELM:
        """
        Parameters
        ----------
        X : array-like
            The input features for training the network.

        y : list, tuple, or np.ndarray
            The target labels for the input data. It must be 1D and will be processed to determine the number of unique labels.
        """
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                self.n_labels, self.classes_ = len(np.unique(y)), np.unique(y)
            # elif y.ndim == 2:
            #     self.n_labels, self.classes_ = y.shape[1], np.arange(y.shape[1])
            else:
                raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        else:
            raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")
        if self.n_labels > 2:
            if self.obj_name in self.CLS_OBJ_LOSSES:
                self.return_prob = True

        network = MultiLayerELM(layer_sizes=self.layer_sizes, act_name=self.act_name, seed=self.seed)
        ohe_scaler = OneHotEncoder(sparse_output=False)
        ohe_scaler.fit(np.reshape(y, (-1, 1)))
        network.obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        network.input_size = X.shape[1]
        return network

    def objective_function(self, solution=None):
        """
        Evaluates the fitness function for classification metric

        Parameters
        ----------
        solution : np.ndarray, default=None

        Returns
        -------
        result: float
            The fitness value
        """
        self.network.decode(solution, self.X_temp, self.y_temp)
        y_pred = self.predict(self.X_temp, return_prob=self.return_prob)
        y1 = self.network.obj_scaler.inverse_transform(self.y_temp)
        loss_train = ClassificationMetric(y1, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train

    def score(self, X, y, method="AS"):
        """
        Return the metric on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        method : str, default="AS"
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        return self._BaseElm__score_cls(X, y, method)

    def scores(self, X, y, list_methods=("AS", "RS")):
        """
        Return the list of metrics on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric
        since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        list_methods : list, default=("AS", "RS")
            You can get all of the metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseElm__scores_cls(X, y, list_methods)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Return the list of performance metrics on the given test data and labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("AS", "RS")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        return self._BaseElm__evaluate_cls(y_true, y_pred, list_metrics)
