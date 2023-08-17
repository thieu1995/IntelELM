============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/intelelm />`_::

   $ pip install intelelm==1.0.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/intelelm.git
   $ cd intelelm
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/intelelm


After installation, you can import IntelELM as any other Python module::

   $ python
   >>> import intelelm
   >>> intelelm.__version__

========
Examples
========

In this section, we will explore the usage of the IntelELM model with the assistance of a dataset. While all the
preprocessing steps mentioned below can be replicated using Scikit-Learn, we have implemented some utility functions
to provide users with convenience and faster usage.


**Combine IntelELM library like a normal library with scikit-learn**::

	### Step 1: Importing the libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import MinMaxScaler, LabelEncoder
	from intelelm import MhaElmRegressor, MhaElmClassifier

	#### Step 2: Reading the dataset
	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:2].values
	y = dataset.iloc[:, 2].values

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

	##### 5.1: Use standard ELM model for regression problem
	regressor = ElmRegressor(hidden_size=10, act_name="relu")
	regressor.fit(X_train, y_train)

	##### 5.2: Use standard ELM model for classification problem
	classifer = ElmClassifier(hidden_size=10, act_name="tanh")
	classifer.fit(X_train, y_train)

	##### 5.3: Use Metaheuristic-based ELM model for regression problem
	print(MhaElmClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaElmClassifier.SUPPORTED_REG_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	regressor = MhaElmRegressor(hidden_size=10, act_name="elu", obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras)
	regressor.fit(X_train, y_train)

	##### 5.4: Use Metaheuristic-based ELM model for classification problem
	print(MhaElmClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	classifier = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="KLDL", optimizer="BaseGA", optimizer_paras=opt_paras)
	classifier.fit(X_train, y_train)

	#### Step 6: Predicting a new result
	y_pred = regressor.predict(X_test)

	y_pred_cls = classifier.predict(X_test)
	y_pred_label = le_y.inverse_transform(y_pred_cls)

	#### Step 7: Calculate metrics using score or scores functions.
	print("Try my AS metric with score function")
	print(regressor.score(data.X_test, data.y_test, method="AS"))

	print("Try my multiple metrics with scores function")
	print(classifier.scores(data.X_test, data.y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))


**Utilities everything that IntelELM provided**::

	### Step 1: Importing the libraries
	from intelelm import ElmRegressor, ElmClassifier, MhaElmRegressor, MhaElmClassifier, get_dataset

	#### Step 2: Reading the dataset
	data = get_dataset("aniso")

	#### Step 3: Next, split dataset into train and test set
	data.split_train_test(test_size=0.2, shuffle=True, random_state=100)

	#### Step 4: Feature Scaling
	data.X_train, scaler_X = data.scale(data.X_train, method="MinMaxScaler", feature_range=(0, 1))
	data.X_test = scaler_X.transform(data.X_test)

	data.y_train, scaler_y = data.encode_label(data.y_train)   # This is for classification problem only
	data.y_test = scaler_y.transform(data.y_test)

	#### Step 5: Fitting ELM-based model to the dataset

	##### 5.1: Use standard ELM model for regression problem
	regressor = ElmRegressor(hidden_size=10, act_name="relu")
	regressor.fit(data.X_train, data.y_train)

	##### 5.2: Use standard ELM model for classification problem
	classifer = ElmClassifier(hidden_size=10, act_name="tanh")
	classifer.fit(data.X_train, data.y_train)

	##### 5.3: Use Metaheuristic-based ELM model for regression problem
	print(MhaElmClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaElmClassifier.SUPPORTED_REG_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	regressor = MhaElmRegressor(hidden_size=10, act_name="elu", obj_name="RMSE", optimizer="BaseGA", optimizer_paras=opt_paras)
	regressor.fit(data.X_train, data.y_train)

	##### 5.4: Use Metaheuristic-based ELM model for classification problem
	print(MhaElmClassifier.SUPPORTED_OPTIMIZERS)
	print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)
	opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
	classifier = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="KLDL", optimizer="BaseGA", optimizer_paras=opt_paras)
	classifier.fit(data.X_train, data.y_train)

	#### Step 6: Predicting a new result
	y_pred = regressor.predict(data.X_test)

	y_pred_cls = classifier.predict(data.X_test)
	y_pred_label = scaler_y.inverse_transform(y_pred_cls)

	#### Step 7: Calculate metrics using score or scores functions.
	print("Try my AS metric with score function")
	print(regressor.score(data.X_test, data.y_test, method="AS"))

	print("Try my multiple metrics with scores function")
	print(classifier.scores(data.X_test, data.y_test, list_methods=["AS", "PS", "F1S", "CEL", "BSL"]))


A real-world dataset contains features that vary in magnitudes, units, and range. We would suggest performing
normalization when the scale of a feature is irrelevant or misleading. Feature Scaling basically helps to normalize
the data within a particular range.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
