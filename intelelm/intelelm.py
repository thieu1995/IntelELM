#!/usr/bin/env python
# Created by "Thieu" at 10:38, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.preprocessing import OneHotEncoder
from intelelm.utils import activation
from intelelm.utils.encoder import ObjectiveScaler
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers
from intelelm.utils.evaluator import get_all_regression_metrics, get_all_classification_metrics


class ELM:
    """Extreme Learning Machine

    This class defines the general ELM model

    Parameters
    ----------
    size_input : int, default=5
        The number of input nodes

    size_hidden : int, default=10
        The number of hidden nodes

    size_output : int, default=1
        The number of output nodes

    act_name : {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish", "hard_swish", "soft_plus", "mish, "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink" }, default='sigmoid'
        Activation function for the hidden layer.
    """
    def __init__(self, size_input=5, size_hidden=10, size_output=1, act_name='sigmoid'):
        self.input_nodes = size_input
        self.hidden_nodes = size_hidden
        self.output_nodes = size_output
        self.size_w1 = self.input_nodes * self.hidden_nodes
        self.size_b = self.hidden_nodes
        self.size_w2 = self.hidden_nodes * self.output_nodes
        self.act_name = act_name
        self.act_func = getattr(activation, self.act_name)
        self.weights = {
            "w1": np.random.rand(self.input_nodes, self.hidden_nodes),
            "b": np.random.rand(self.hidden_nodes),
            "w2": np.random.rand(self.hidden_nodes, self.output_nodes)
        }

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
            Returns a trained ELM model.
        """
        H = self.act_func(X @ self.weights["w1"] + self.weights["b"])
        self.weights["w2"] = np.linalg.pinv(H) @ y
        return self

    def predict(self, X):
        """Predict using the Extreme Learning Machine model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        H = self.act_func(X @ self.weights["w1"] + self.weights["b"])
        y_pred = H @ self.weights["w2"]
        return y_pred

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights_size(self):
        return np.sum([item.size for item in self.weights.values()])

    def update_weights_from_solution(self, solution, X, y):
        w1 = np.reshape(solution[:self.size_w1], (self.input_nodes, self.hidden_nodes))
        b = np.reshape(solution[self.size_w1:self.size_w1 + self.size_b], (-1, self.hidden_nodes))
        H = self.act_func(X @ self.weights["w1"] + self.weights["b"])
        w2 = np.linalg.pinv(H) @ y
        self.weights = {"w1": w1, "b": b, "w2": w2}


class BaseElm(BaseEstimator):
    """
    Defines the most general class for ELM network that inherits the BaseEstimator class of Scikit-Learn library.

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish", "hard_swish", "soft_plus", "mish, "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink" }, default='sigmoid'
        Activation function for the hidden layer.

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    pathsave : str, default="history"
        The folder to save the results
    """
    def __init__(self, hidden_size=10, act_name="elu", pathsave="history", verbose=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.act_name = act_name
        self.pathsave = pathsave
        self.verbose = verbose
        self.weights = {}
        self.loss_train = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class BaseMhaElm(BaseElm):
    """
    Defines the most general class for Metaheuristic-based ELM model that inherits the BaseELM class

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish", "hard_swish", "soft_plus", "mish, "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink" }, default='sigmoid'
        Activation function for the hidden layer.

    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    pathsave : str, default="history"
        The folder to save the results

    verbose : bool, default=True
        Whether to print progress messages to stdout.
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers().keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, hidden_size=10, act_name="elu", obj_name=None, optimizer="BaseGA", optimizer_paras=None, pathsave="history", verbose=True):
        super().__init__(hidden_size=hidden_size, act_name=act_name, pathsave=pathsave, verbose=verbose)
        self.obj_name = obj_name
        self.optimizer_paras = optimizer_paras
        self.optimizer = self._set_optimizer(optimizer, optimizer_paras)
        self.network, self.obj_scaler = None, None

    def _set_optimizer(self, optimizer=None, optimizer_paras=None):
        if type(optimizer) is str:
            opt_class = get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                return opt_class(**optimizer_paras)
            else:
                return opt_class(epoch=500, pop_size=50)
        elif isinstance(optimizer, Optimizer):
            if type(optimizer_paras) is dict:
                return optimizer.set_parameters(optimizer_paras)
            return optimizer
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def _create_network(self, X, y):
        return None, None

    def _get_history_loss(self, optimizer=None):
        list_global_best = optimizer.history.list_global_best
        # 2D array / matrix 2D
        global_obj_list = np.array([agent[1][-1] for agent in list_global_best])
        # Make each obj_list as a element in array for drawing
        return global_obj_list[:, 0]

    def fitness_function(self, solution=None):
        pass

    def fit(self, X, y):
        self.network, self.obj_scaler = self._create_network(X, y)
        y_scaled = self.obj_scaler.transform(y)
        self.X_temp, self.y_temp = X, y_scaled
        self.size_w1 = X.shape[1] * self.hidden_size
        self.size_b = self.hidden_size
        self.size_w2 = self.hidden_size * 1
        problem_size = self.size_w1 + self.size_b
        lb = [-1, ] * problem_size
        ub = [1, ] * problem_size
        log_to = "console" if self.verbose else "None"
        if self.obj_name is None:
            raise ValueError("obj_name can't be None")
        else:
            if self.obj_name in self.SUPPORTED_REG_OBJECTIVES.keys():
                minmax = self.SUPPORTED_REG_OBJECTIVES[self.obj_name]
            elif self.obj_name in self.SUPPORTED_CLS_OBJECTIVES.keys():
                minmax = self.SUPPORTED_CLS_OBJECTIVES[self.obj_name]
            else:
                raise ValueError("obj_name is not supported. Please check the library: permetrics to see the supported objective function.")
        problem = {
            "fit_func": self.fitness_function,
            "lb": lb,
            "ub": ub,
            "minmax": minmax,
            "log_to": log_to,
            "save_population": False,
        }
        self.solution, self.best_fit = self.optimizer.solve(problem)
        self.loss_train = self._get_history_loss(optimizer=self.optimizer)
        return self

    def predict(self, X, return_prob=False):
        """
        Inherit the predict function from BaseElm class, with 1 more parameter `return_prob`.

        Parameters
        ----------
        X :
        return_prob : bool, default=False
            It is used for classification problem:

            - If True, the returned results are the probability for each sample
            - If False, the returned results are the predicted labels
        """
        pred = self.network.predict(X)
        if return_prob:
            return pred
        return self.obj_scaler.inverse_transform(pred)


class MhaElmRegressor(BaseMhaElm, RegressorMixin):
    """
    Defines the general class of Metaheuristic-based ELM model for Regression problems that inherit the BaseMhaElm and RegressorMixin classes.

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish", "hard_swish", "soft_plus", "mish, "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink" }, default='sigmoid'
        Activation function for the hidden layer.

    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    pathsave : str, default="history"
        The folder to save the results

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    Examples
    --------
    >>> from intelelm import MhaElmRegressor, Data
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    >>> model = MhaElmRegressor(hidden_size=10, act_name="elu", obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    """

    def __init__(self, hidden_size=10, act_name="elu", obj_name=None, optimizer="BaseGA", optimizer_paras=None, pathsave="history", verbose=True):
        super().__init__(hidden_size=hidden_size, act_name=act_name, obj_name=obj_name, optimizer=optimizer,
                            optimizer_paras=optimizer_paras, pathsave=pathsave, verbose=verbose)

    def _create_network(self, X, y):
        dim = 1
        if y.ndim > 1:
            y = np.squeeze(y)
            dim = y.ndim
        obj_scaler = ObjectiveScaler(obj_name="self", ohe_scaler=None)
        network = ELM(size_input=X.shape[1], size_hidden=self.hidden_size, size_output=dim, act_name=self.act_name)
        return network, obj_scaler

    def fitness_function(self, solution=None):
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
        self.network.update_weights_from_solution(solution, self.X_temp, self.y_temp)
        y_pred = self.network.predict(self.X_temp)
        loss_train = RegressionMetric(self.y_temp, y_pred, decimal=8).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train


class MhaElmClassifier(BaseMhaElm, ClassifierMixin):
    """
    Defines the general class of Metaheuristic-based ELM model for Classification problems that inherit the BaseMhaElm and ClassifierMixin classes.

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid",
        "swish", "hard_swish", "soft_plus", "mish, "soft_sign", "tanh_shrink", "soft_shrink", "hard_shrink" }, default='sigmoid'
        Activation function for the hidden layer.

    obj_name : None or str, default=None
        The name of objective for the problem, also depend on the problem is classification and regression.

    optimizer : str or instance of Optimizer class (from Mealpy library), default = "BaseGA"
        The Metaheuristic Algorithm that use to solve the feature selection problem.
        Current supported list, please check it here: https://github.com/thieu1995/mealpy.
        If a custom optimizer is passed, make sure it is an instance of `Optimizer` class.

    optimizer_paras : None or dict of parameter, default=None
        The parameter for the `optimizer` object.
        If `None`, the default parameters of optimizer is used (defined in https://github.com/thieu1995/mealpy.)
        If `dict` is passed, make sure it has at least `epoch` and `pop_size` parameters.

    pathsave : str, default="history"
        The folder to save the results

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    Examples
    --------
    >>> from intelelm import Data, MhaElmClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> data = Data(X, y)
    >>> data.split_train_test(test_size=0.2, random_state=1)
    >>> opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    >>> print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)
    {'PS': 'max', 'NPV': 'max', 'RS': 'max', ...., 'KLDL': 'min', 'BSL': 'min'}
    >>> model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="BSL", optimizer="BaseGA", optimizer_paras=opt_paras)
    >>> model.fit(data.X_train, data.y_train)
    >>> pred = model.predict(data.X_test)
    >>> print(pred)
    array([1, 0, 1, 0, 1])
    """

    OBJ_LOSSES = ["CEL", "HL", "KLDL", "BSL"]

    def __init__(self, hidden_size=10, act_name="elu", obj_name=None, optimizer="BaseGA", optimizer_paras=None, pathsave="history", verbose=True):
        super().__init__(hidden_size=hidden_size, act_name=act_name, obj_name=obj_name, optimizer=optimizer,
                         optimizer_paras=optimizer_paras, pathsave=pathsave, verbose=verbose)
        self.return_prob = False

    def _check_y(self, y):
        if type(y) in (list, tuple, np.ndarray):
            y = np.squeeze(np.asarray(y))
            if y.ndim == 1:
                return len(np.unique(y))
            raise TypeError("Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on.")
        raise TypeError("Invalid y array type, it should be list, tuple or np.ndarray")

    def _create_network(self, X, y):
        self.n_labels = self._check_y(y)
        if self.n_labels > 2:
            if self.obj_name in self.OBJ_LOSSES:
                self.return_prob = True
        ohe_scaler = OneHotEncoder(sparse=False)
        ohe_scaler.fit(np.reshape(y, (-1, 1)))
        obj_scaler = ObjectiveScaler(obj_name="softmax", ohe_scaler=ohe_scaler)
        network = ELM(size_input=X.shape[1], size_hidden=self.hidden_size, size_output=self.n_labels, act_name=self.act_name)
        return network, obj_scaler

    def fitness_function(self, solution=None):
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
        self.network.update_weights_from_solution(solution, self.X_temp, self.y_temp)
        y_pred = self.predict(self.X_temp, return_prob=self.return_prob)
        y1 = self.obj_scaler.inverse_transform(self.y_temp)
        loss_train = ClassificationMetric(y1, y_pred, decimal=8).get_metric_by_name(self.obj_name)[self.obj_name]
        return loss_train
