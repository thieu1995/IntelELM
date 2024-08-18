#!/usr/bin/env python
# Created by "Thieu" at 06:52, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import validation_curve
from permetrics.regression import RegressionMetric
from permetrics.classification import ClassificationMetric
from sklearn.metrics import make_scorer


def get_metrics(problem, y_true, y_pred, metrics=None, testcase="test"):
    """
    Calculates metrics for regression or classification tasks.

    This function takes the true labels (y_true), predicted labels (y_pred), problem type
    (regression or classification), a dictionary or list of metrics to calculate, and an
    optional test case name. It returns a dictionary containing the calculated metrics with
    descriptive names.

    Args:
        problem (str): The type of problem, either "regression" or "classification".
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.
        metrics (dict or list, optional): A dictionary or list of metrics to calculate. Defaults to None.
        testcase (str, optional): An optional test case name to prepend to the metric names. Defaults to "test".

    Returns:
        dict: A dictionary containing the calculated metrics with descriptive names.

    Raises:
        ValueError: If the `metrics` parameter is not a list or dictionary.
    """
    if problem == "regression":
        evaluator = RegressionMetric(y_true, y_pred)
        paras = [{}, ] * len(metrics)   # Create empty parameter lists for metrics
    else:
        evaluator = ClassificationMetric(y_true, y_pred)
        paras = [{"average": "weighted"}, ] * len(metrics)  # Set default parameters for classification metrics
    if type(metrics) is dict:
        result = evaluator.get_metrics_by_dict(metrics)     # Calculate metrics using a dictionary
    elif type(metrics) in (tuple, list):
        result = evaluator.get_metrics_by_list_names(metrics, paras)    # Calculate metrics using a list of names and parameters
    else:
        raise ValueError("metrics parameter should be a list or dict")
    final_result = {}
    for key, value in result.items():
        if testcase is None or testcase == "":
            final_result[f"{key}"] = value              # Add metric name without test case prefix if not provided
        else:
            final_result[f"{key}_{testcase}"] = value   # Add metric name with test case prefix
    return final_result


def get_all_regression_metrics():
    """
    Gets a dictionary of all supported regression metrics.

    This function returns a dictionary where keys are metric names and values are their optimization types ("min" or "max").

    Returns:
        dict: A dictionary containing all supported regression metrics.
    """
    UNUSED_METRICS = ("RE", "RB", "AE", "SE", "SLE")        # List of unused metrics
    dict_results = {}
    for key, value in RegressionMetric.SUPPORT.items():
        if (key not in UNUSED_METRICS) and (value["type"] in ("min", "max")):
            dict_results[key] = value["type"]
    return dict_results


def get_all_classification_metrics():
    """
    Gets a dictionary of all supported classification metrics.

    This function returns a dictionary where keys are metric names and values are their optimization types ("min" or "max").

    Returns:
        dict: A dictionary containing all supported classification metrics.
    """
    dict_results = {}
    for key, value in ClassificationMetric.SUPPORT.items():
        if value["type"] in ("min", "max"):
            dict_results[key] = value["type"]   # Add supported metrics and their optimization types
    return dict_results


def get_metric_sklearn(task="classification", metric_names=None):
    """
      Creates a dictionary of scorers for scikit-learn cross-validation.

      This function takes the task type (classification or regression) and a list of metric names.
      It creates an appropriate metrics instance (ClassificationMetric or RegressionMetric) and iterates
      through the provided metric names. For each metric name, it checks if it exists in the metrics
      instance and retrieves the corresponding method. Finally, it uses `make_scorer` to convert the
      method to a scorer and adds it to a dictionary.

      Args:
          task (str, optional): The task type, either "classification" or "regression". Defaults to "classification".
          metric_names (list, optional): A list of metric names. Defaults to None.

      Returns:
          dict: A dictionary of scorers for scikit-learn cross-validation.
      """
    if task == "classification":
        met = ClassificationMetric()
    else:
        met = RegressionMetric()
    # Initialize an empty dictionary to hold scorers
    scorers = {}
    # Loop through metric names, dynamically create scorers, and add them to the dictionary
    for metric_name in metric_names:
        # Get the method from the metrics instance
        if hasattr(met, metric_name):
            metric_method = getattr(met, metric_name.upper())
        else:
            continue
        # Convert the method to a scorer using make_scorer
        if met.SUPPORT[metric_name]["type"] == "min":
            greater_is_better = False
        else:
            greater_is_better = True
        scorers[metric_name] = make_scorer(metric_method, greater_is_better=greater_is_better)
    # Now, you can use this scorers dictionary
    return scorers
