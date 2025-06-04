============
Installation
============

There are so many ways to install our library. For example:

* Install from the `PyPI release <https://pypi.python.org/pypi/intelelm />`_::

   $ pip install intelelm==1.3.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/intelelm.git
   $ cd intelelm
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/intelelm


After installation, you can check the version of IntelELM::

   $ python
   >>> import intelelm
   >>> intelelm.__version__

=========
Tutorials
=========

-------------------------
1) Getting started in 30s
-------------------------

In the example below, we will apply the traditional ELM model to the diabetes prediction problem. This dataset is already available in our library. The
process consists of the following steps:
	* Import libraries
	* Load and split dataset
	* Scale dataset
	* Define the model
	* Train the model
	* Test the model
	* Evaluate the model

.. code-block:: python

	## Import libraries
	import numpy as np
	from intelelm import get_dataset, ElmRegressor

	## Load dataset
	data = get_dataset("diabetes")
	data.split_train_test(test_size=0.2, random_state=2)
	print(data.X_train.shape, data.X_test.shape)

	## Scale dataset
	data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=('minmax', ))
	data.X_test = scaler_X.transform(data.X_test)

	data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=('minmax', ))
	data.y_test = scaler_y.transform(np.reshape(data.y_test, (-1, 1)))

	## Define the model
	model = ElmRegressor(layer_sizes=(10, ), act_name="elu", seed=42)

	## Test the model
	model.fit(data.X_train, data.y_train)

	## Test the model
	y_pred = model.predict(data.X_test)

	## Evaluate the model
	print(model.score(data.X_test, data.y_test, method="RMSE"))
	print(model.scores(data.X_test, data.y_test, list_methods=("RMSE", "MAPE")))
	print(model.evaluate(y_true=data.y_test, y_pred=y_pred, list_metrics=("MAPE", "R2", "NSE")))


As you can see, it is very similar to any other Estimator model in the Scikit-Learn library. They only differ in the model definition part.
In the provided example, we used the ElmRegressor from the library, which is specifically designed for Extreme Learning Machines. However, the overall
workflow follows the familiar pattern of loading data, preprocessing, training, and evaluating the model.


-------------------
2) Model Definition
-------------------

**Metaheuristic Optimization-based ELM model**
If you want to use the Whale Optimization-based ELM (WO-ELM) model, you can change the model definition like this:

.. code-block:: python

	from intelelm import MhaElmRegressor

	opt_paras = {"name": "WOA", "epoch": 100, "pop_size": 30}
	model = MhaElmRegressor(layer_sizes=(10, ), act_name="elu", obj_name="MSE",
			optim="OriginalWOA", optim_params=opt_paras, verbose=False, seed=42)

In the example above, I had to import the MhaElmRegressor class. This is the class that contains all Metaheuristics-based ELM models for regression problems.
Then, I defined parameters for the Whale Optimization algorithm. And I defined parameters for the Whale Optimization-based ELM model.


**What about hybrid model for Classification problem**

In case you want to use the model for a classification problem, you need to import either the ElmClassifier class (this is the traditional ELM model) or the
MhaElmClassifier class (these are hybrid models combining metaheuristics algorithms and ELM networks).

.. code-block:: python

	from intelelm import ElmClassifier

	model = ElmClassifier(layer_sizes=(10, ), act_name="elu", seed=42)



.. code-block:: python

	from intelelm import ElmClassifier

	opt_paras = {"name": "GA", "epoch": 100, "pop_size": 30}
	model = MhaElmClassifier(layer_sizes=(10, ), act_name="elu", obj_name="BSL",
			optim="BaseGA", optim_params=opt_paras, verbose=False, seed=42)


-------------------
3) Data Preparation
-------------------


If you want to use your own data, it's straightforward. You won't need to load the data into our Data class. However, you'll need to use the Scikit-Learn
library to split and scale the data.


.. code-block:: python

	### Step 1: Importing the libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import MinMaxScaler, LabelEncoder
	from intelelm import MhaElmClassifier

	#### Step 2: Reading the dataset
	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:2].values         # This is features
	y = dataset.iloc[:, 2].values           # This is output

	#### Step 3: Next, split dataset into train and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=100)

	#### Step 4: Feature Scaling
	scaler_X = MinMaxScaler()
	scaler_X.fit(X_train)
	X_train = scaler_X.transform(X_train)
	X_test = scaler_X.transform(X_test)

	le_y = LabelEncoder()       # This is for classification problem only
	le_y.fit(y)
	y_train = le_y.transform(y_train)
	y_test = le_y.transform(y_test)

	#### Step 5: Fitting ELM-based model to the dataset

	##### 5.1: Use standard ELM model for classification problem
	classifer = ElmClassifier(layer_sizes=(10, ), act_name="tanh")

	##### 5.2: Use Metaheuristic-based ELM model for classification problem
	print(MhaElmClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	classifier = MhaElmClassifier(layer_sizes=(10, ), act_name="elu", obj_name="KLDL", optim="BaseGA", optim_params=opt_paras, seed=42)

	#### Step 6: Traint the model
	classifer.fit(X_train, y_train)

	#### Step 7: Predicting a new result
	y_pred = regressor.predict(X_test)

	y_pred_cls = classifier.predict(X_test)
	y_pred_label = le_y.inverse_transform(y_pred_cls)

	#### Step 8: Calculate metrics using score or scores functions.
	print("Try my AS metric with score function")
	print(regressor.score(data.X_test, data.y_test, method="AS"))

	print("Try my multiple metrics with scores function")
	print(classifier.scores(data.X_test, data.y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))


A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize
the data within a particular range.


---------------------------
4) Scikit-Learn Integration
---------------------------

There's no need to delve further into this issue. The classes in the IntelELM library inherit from the BaseEstimator class from the Scikit-Learn library.
Therefore, the features provided by the Scikit-Learn library can be utilized by the classes in the IntelELM library.


In the example below, we use the Whale Optimization-based ELM model as the base model for the recursive feature selection method for feature selection problem.

.. code-block:: python

	# import necessary class, modules, and functions
	from intelelm import Data, MhaElmRegressor
	from sklearn.feature_selection import RFE

	# load X features and y label from file
	X, y = load_my_data() # Assumption that this is loading data function

	# create data object
	data = Data(X, y)

	# create model and selector
	opt_paras = {"name": "GA", "epoch": 100, "pop_size": 30}
	model = MhaElmRegressor(layer_sizes=(10, ), act_name="relu", obj_name="MSE",
			optim="BaseGA", optim_params=opt_paras, verbose=False, seed=42)

	selector = RFE(estimator=model)
	selector.fit(X_train, y_train)

	# get the final dataset
	data.X_train = data.X_train[selector.support_]
	data.X_test = data.X_test[selector.support_]
	print(f'Ranking of features from Recursive FS: {selector.ranking_}')


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
