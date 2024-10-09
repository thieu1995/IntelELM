# Version 1.2.0

+ Rename `ELM` class to `MultiLayerELM` class. This new class can be used to define deep ELM network
  by `layer_sizes` parameter. 
+ Add `AutomatedMhaElmTuner` class that can be used to perform hyperparameter tuning for MhaElm models using 
  either GridSearchCV or RandomizedSearchCV. Provides an interface for fitting and predicting using the best found model.
+ Add `AutomatedMhaElmComparator` class that automatic compare different MhaElm models based on provided optimizer 
  configurations. It provides methods for cross-validation and train-test split evaluation.
+ Update docs, examples, and tests.

---------------------------------------------------------------------

# Version 1.1.1

+ Update seed value in all 4 classes to ensure reproducibility of your results
+ Add mode, n_workers, and termination parameter in model.fit() of MhaElmRegressor and MhaElmClassifier classes
  + These parameters are derived from Mealpy library
  + With mode parameter, you can speed your training model
  + With n_workers, you can set the number of threads or CPUs to speed up the training process
  + With termination, you can set early stopping strategy for your model.
+ Update docs, examples, and tests.

---------------------------------------------------------------------

# Version 1.1.0

+ Update core modules to fit upgraded version of Mealpy>=3.0.1, PerMetrics>=2.0.0, Scikit-Learn>=1.2.1
+ IntelELM no longer support Python 3.7. Only support Python >= 3.8
+ Update docs and add examples

---------------------------------------------------------------------

# Version 1.0.3

+ Fix bug lb and ub in BaseMhaElm class
+ Update docs and add example

---------------------------------------------------------------------

# Version 1.0.2

+ Fix bug in DataTransformer class
+ Fix bug in LabelEncoder class
+ Add more activation functions 
+ Update documents, examples

---------------------------------------------------------------------

# Version 1.0.1

+ Add "evaluate" function to all Estimators (ElmRegressor, ElmClassifier, MhaElmRegressor, MhaElmClassifier)
+ **Add new module "scaler"**
+ Our scaler can be utilized with multiple methods.
+ Add "save_loss_train" and "save_metrics" functions to all Estimators
+ Add "save_model" and "load_model" functions to all Estimators
+ Add "save_y_predicted" function to all Estimators
+ Update all examples and documents

---------------------------------------------------------------------


# Version 1.0.0 

+ Add supported information for each classes.
+ Restructure intelelm module to based_elm module and model subpackage that includes mha_elm and standard_elm modules.
+ Add traditional/standard ELM models (ElmRegressor and ElmClassifier classes) to standard_elm module.
+ Add examples and tests for traditional models
+ Add score and scores functions to all classes.
+ Fix bug calculate metrics and objective in ELM-based models.
+ Add examples with real-world datasets and examples with GridsearchCV to tune hyper-parameters of ELM-based models.
+ Add documents

---------------------------------------------------------------------

# Version 0.1.0 (First version)

+ Add infors (CODE_OF_CONDUCT.md, MANIFEST.in, LICENSE, README.md, requirements.txt, CITATION.cff)
+ Add supported classification and regression datasets
+ Add util modules (data_loader, validator, evaluator, encoder, activation)
+ Add MhaElmRegressor and MhaElmClassifier classes
+ Add publish workflows
+ Add examples and tests folders
