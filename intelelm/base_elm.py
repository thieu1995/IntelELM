#!/usr/bin/env python
# Created by "Thieu" at 09:48, 17/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import BaseEstimator
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar
from intelelm.utils import activation, validator
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

    act_name : {"relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid",
        "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink",
        "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" }, default='sigmoid'
        Activation function for the hidden layer.

    seed: int, default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.
    """
    def __init__(self, size_input=5, size_hidden=10, size_output=1, act_name='sigmoid', seed=None):
        self.generator = np.random.default_rng(seed)
        self.input_nodes = size_input
        self.hidden_nodes = size_hidden
        self.output_nodes = size_output
        self.size_w1 = self.input_nodes * self.hidden_nodes
        self.size_b = self.hidden_nodes
        self.size_w2 = self.hidden_nodes * self.output_nodes
        self.act_name = act_name
        self.act_func = getattr(activation, self.act_name)
        self.weights = {
            "w1": self.generator.random((self.input_nodes, self.hidden_nodes)),
            "b": self.generator.random(self.hidden_nodes),
            "w2": self.generator.random((self.hidden_nodes, self.output_nodes))
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
        H = self.act_func(np.dot(X, self.weights["w1"]) + self.weights["b"])
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
        H = self.act_func(np.dot(X, self.weights["w1"]) + self.weights["b"])
        y_pred = np.dot(H, self.weights["w2"])
        return y_pred

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights_size(self):
        return np.sum([item.size for item in self.weights.values()])

    def update_weights_from_solution(self, solution, X, y):
        w1 = np.reshape(solution[:self.size_w1], (self.input_nodes, self.hidden_nodes))
        b = np.reshape(solution[self.size_w1:self.size_w1 + self.size_b], self.hidden_nodes)
        H = self.act_func(np.dot(X, w1) + b)
        w2 = np.dot(np.linalg.pinv(H), y)
        self.set_weights({"w1": w1, "b": b, "w2": w2})


class BaseElm(BaseEstimator):
    """
    Defines the most general class for ELM network that inherits the BaseEstimator class of Scikit-Learn library.

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid",
        "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink",
        "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" }, default='sigmoid'
        Activation function for the hidden layer.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()
    CLS_OBJ_LOSSES = None

    def __init__(self, hidden_size=10, act_name="elu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.act_name = act_name
        self.weights = {}
        self.network, self.obj_scaler, self.loss_train = None, None, None
        self.n_labels, self.obj_scaler = None, None 

    @staticmethod
    def _check_method(method=None, list_supported_methods=None):
        if type(method) is str:
            return validator.check_str("method", method, list_supported_methods)
        else:
            raise ValueError(f"method should be a string and belongs to {list_supported_methods}")

    def create_network(self, X, y):
        return None, None

    def fit(self, X, y):
        self.network, self.obj_scaler = self.create_network(X, y)
        y_scaled = self.obj_scaler.transform(y)
        self.network.fit(X, y_scaled)
        return self

    def predict(self, X, return_prob=False):
        """
        Inherit the predict function from BaseElm class, with 1 more parameter `return_prob`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        return_prob : bool, default=False
            It is used for classification problem:

            - If True, the returned results are the probability for each sample
            - If False, the returned results are the predicted labels
        """
        pred = self.network.predict(X)
        if return_prob:
            return pred
        return self.obj_scaler.inverse_transform(pred)

    def __evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def __evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)
    
    def __score_reg(self, X, y, method="RMSE"):
        method = self._check_method(method, list(self.SUPPORTED_REG_METRICS.keys()))
        y_pred = self.network.predict(X)
        return RegressionMetric(y, y_pred).get_metric_by_name(method)[method]
    
    def __scores_reg(self, X, y, list_methods=("MSE", "MAE")):
        y_pred = self.network.predict(X)
        return self.__evaluate_reg(y_true=y, y_pred=y_pred, list_metrics=list_methods)

    def __score_cls(self, X, y, method="AS"):
        method = self._check_method(method, list(self.SUPPORTED_CLS_METRICS.keys()))
        return_prob = False
        if self.n_labels > 2:
            if method in self.CLS_OBJ_LOSSES:
                return_prob = True
        y_pred = self.predict(X, return_prob=return_prob)
        cm = ClassificationMetric(y_true=y, y_pred=y_pred)
        return cm.get_metric_by_name(method)[method]

    def __scores_cls(self, X, y, list_methods=("AS", "RS")):
        list_errors = list(set(list_methods) & set(self.CLS_OBJ_LOSSES))
        list_scores = list((set(self.SUPPORTED_CLS_METRICS.keys()) - set(self.CLS_OBJ_LOSSES)) & set(list_methods))
        t1 = {}
        if len(list_errors) > 0:
            return_prob = False
            if self.n_labels > 2:
                return_prob = True
            y_pred = self.predict(X, return_prob=return_prob)
            t1 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_errors)
        y_pred = self.predict(X, return_prob=False)
        t2 = self.__evaluate_cls(y_true=y, y_pred=y_pred, list_metrics=list_scores)
        return {**t2, **t1}

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """Return the list of performance metrics of the prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def score(self, X, y, method=None):
        """Return the metric of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        method : str, default="RMSE"
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        result : float
            The result of selected metric
        """
        pass

    def scores(self, X, y, list_methods=None):
        """Return the list of metrics of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        list_methods : list, default=("MSE", "MAE")
            You can get metrics from Permetrics library: https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            The results of the list metrics
        """
        pass

    def save_loss_train(self, save_path="history", filename="loss.csv"):
        """
        Save the loss (convergence) during the training process to csv file.

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} model doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to csv file

        Parameters
        ----------
        y_true : ground truth data
        y_pred : predicted output
        list_metrics : list of evaluation metrics
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save the predicted results to csv file

        Parameters
        ----------
        X : The features data, nd.ndarray
        y_true : The ground truth data
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X, return_prob=False)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="model.pkl"):
        """
        Save model to pickle file

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".pkl" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="model.pkl"):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))


class BaseMhaElm(BaseElm):
    """
    Defines the most general class for Metaheuristic-based ELM model that inherits the BaseELM class

    Parameters
    ----------
    hidden_size : int, default=10
        The number of hidden nodes

    act_name : {"relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu", "rrelu", "tanh", "hard_tanh", "sigmoid",
        "hard_sigmoid", "log_sigmoid", "silu", "swish", "hard_swish", "soft_plus", "mish", "soft_sign", "tanh_shrink",
        "soft_shrink", "hard_shrink", "softmin", "softmax", "log_softmax" }, default='sigmoid'
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

    verbose : bool, default=True
        Whether to print progress messages to stdout.

    seed: int, default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers().keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, hidden_size=10, act_name="elu", obj_name=None, optimizer="BaseGA", optimizer_paras=None, verbose=True, seed=None):
        super().__init__(hidden_size=hidden_size, act_name=act_name)
        self.obj_name = obj_name
        if optimizer_paras is None:
            optimizer_paras = {"epoch": 500, "pop_size": 20}
        self.optimizer_paras = optimizer_paras
        self.optimizer = optimizer
        self.verbose = verbose
        self.seed = seed
        self.network, self.obj_scaler = None, None
        self.obj_weights = None

    def get_name(self):
        if type(self.optimizer) is str:
            return f"{self.optimizer_paras}-ELM"
        return f"{self.optimizer.name}-ELM"

    def set_params(self, **params):
        # Handle nested parameters for the optimizer
        optimizer_params = {k.split('__')[1]: v for k, v in params.items() if k.startswith('optimizer_paras__')}

        if optimizer_params:
            self.optimizer_paras.update(optimizer_params)

        # Pass non-optimizer parameters to the parent class set_params
        super_params = {k: v for k, v in params.items() if not k.startswith('optimizer_paras__')}
        super().set_params(**super_params)

        return self

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_optimizer_paras(self, optimizer_paras):
        self.optimizer_paras = optimizer_paras

    def set_optimizer_object(self, optimizer=None, optimizer_paras=None):
        if type(optimizer) is str:
            opt_class = get_optimizer_by_name(optimizer)
            if type(optimizer_paras) is dict:
                self.optimizer_paras = optimizer_paras
                self.optimizer = opt_class(**optimizer_paras)
            else:
                raise TypeError(f"optimizer_paras is a dictionary contains the hyper-parameter of optimizer in Mealpy library.")
        elif isinstance(optimizer, Optimizer):
            if type(optimizer_paras) is dict:
                self.optimizer_paras = optimizer_paras
                optimizer.set_parameters(optimizer_paras)
            self.optimizer = optimizer
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def set_seed(self, seed):
        self.seed = seed

    def _get_history_loss(self, optimizer=None):
        list_global_best = optimizer.history.list_global_best
        # 2D array / matrix 2D
        global_obj_list = np.array([agent.target.objectives for agent in list_global_best])
        # Make each obj_list as an element in array for drawing
        return global_obj_list[:, 0]

    def fitness_function(self, solution=None):
        pass

    def fit(self, X, y, lb=(-10.0, ), ub=(10.0, ), mode="single", n_workers=None, termination=None, save_population=False):
        """
        Parameters
        ----------
        X : The features data, np.ndarray
        y : The ground truth data
        lb : The lower bound for decision variables in optimization problem (The weights and biases of network)
        ub : The upper bound for decision variables in optimization problem (The weights and biases of network)
        mode: Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                * 'process': The parallel mode with multiple cores run the tasks
                * 'thread': The parallel mode with multiple threads run the tasks
                * 'swarm': The sequential mode that no effect on updating phase of other agents
                * 'single': The sequential mode that effect on updating phase of other agents, this is default mode

        n_workers: The number of workers (cores or threads) to do the tasks (effect only on parallel mode)
        termination: The termination dictionary or an instance of Termination class in Mealpy library
        save_population : Save the population of search agents (Don't set it to True when you don't know how to use it)
        """
        self.network, self.obj_scaler = self.create_network(X, y)
        y_scaled = self.obj_scaler.transform(y)
        self.X_temp, self.y_temp = X, y_scaled
        problem_size = X.shape[1] * self.hidden_size + self.hidden_size
        if type(lb) in (list, tuple, np.ndarray) and type(ub) in (list, tuple, np.ndarray):
            if len(lb) == len(ub):
                if len(lb) == 1:
                    lb = np.array(lb * problem_size, dtype=float)
                    ub = np.array(ub * problem_size, dtype=float)
                elif len(lb) != problem_size:
                    raise ValueError(f"Invalid lb and ub. Their length should be equal to 1 or problem_size.")
            else:
                raise ValueError(f"Invalid lb and ub. They should have the same length.")
        elif type(lb) in (int, float) and type(ub) in (int, float):
            lb = (float(lb), ) * problem_size
            ub = (float(ub), ) * problem_size
        else:
            raise ValueError(f"Invalid lb and ub. They should be a number of list/tuple/np.ndarray with size equal to problem_size")
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
            "obj_func": self.fitness_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": minmax,
            "log_to": log_to,
            "save_population": save_population,
            "obj_weights": self.obj_weights
        }
        self.set_optimizer_object(self.optimizer, self.optimizer_paras)
        g_best = self.optimizer.solve(problem, mode=mode, n_workers=n_workers, termination=termination, seed=self.seed)
        self.solution, self.best_fit = g_best.solution, g_best.target.fitness
        self.network.update_weights_from_solution(self.solution, self.X_temp, self.y_temp)
        self.loss_train = self._get_history_loss(optimizer=self.optimizer)
        return self
