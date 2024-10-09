.. IntelELM documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to IntelELM's documentation!
====================================

.. image:: https://img.shields.io/badge/release-1.2.0-yellow.svg
   :target: https://github.com/thieu1995/intelelm/releases

.. image:: https://img.shields.io/pypi/wheel/gensim.svg
   :target: https://pypi.python.org/pypi/intelelm

.. image:: https://badge.fury.io/py/intelelm.svg
   :target: https://badge.fury.io/py/intelelm

.. image:: https://img.shields.io/pypi/pyversions/intelelm.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/status/intelelm.svg
   :target: https://img.shields.io/pypi/status/intelelm.svg

.. image:: https://img.shields.io/pypi/dm/intelelm.svg
   :target: https://img.shields.io/pypi/dm/intelelm.svg

.. image:: https://github.com/thieu1995/intelelm/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/intelelm/actions/workflows/publish-package.yaml

.. image:: https://static.pepy.tech/badge/intelelm
   :target: https://pepy.tech/project/intelelm

.. image:: https://img.shields.io/github/release-date/thieu1995/intelelm.svg
   :target: https://img.shields.io/github/release-date/thieu1995/intelelm.svg

.. image:: https://readthedocs.org/projects/intelelm/badge/?version=latest
   :target: https://intelelm.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/github/contributors/thieu1995/intelelm.svg
   :target: https://img.shields.io/github/contributors/thieu1995/intelelm.svg

.. image:: https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?
   :target: https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8249045.svg
   :target: https://doi.org/10.5281/zenodo.8249045

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


**IntelELM: A Python Framework for Intelligent Metaheuristic-based Extreme Learning Machine**

IntelELM (Intelligent Metaheuristic-based Extreme Learning Machine) is a Python library that implements a framework
for training Extreme Learning Machine (ELM) networks using Metaheuristic Algorithms. It provides a comparable
alternative to the traditional ELM network and is compatible with the Scikit-Learn library. With IntelELM, you can
perform searches and hyperparameter tuning using the functionalities provided by the Scikit-Learn library.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Estimator**: ElmRegressor, ElmClassifier, MhaElmRegressor, MhaElmClassifier, AutomatedMhaElmTuner, AutomatedMhaElmComparator
* **Total Optimization-based ELM Regression**: > 200 Models
* **Total Optimization-based ELM Classification**: > 200 Models
* **Supported datasets**: 54 (47 classifications and 7 regressions)
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://intelelm.readthedocs.io/en/latest/
* **Python versions:** >= 3.7.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics

.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/intelelm.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
