#!/usr/bin/env python
# Created by "Thieu" at 09:49, 25/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from intelelm import MhaElmClassifier, Data
from sklearn.datasets import make_classification


# Create a multi-class classification dataset with 4 classes
X, y = make_classification(
    n_samples=300,  # Total number of data points
    n_features=7,  # Number of features
    n_informative=3,  # Number of informative features
    n_redundant=0,  # Number of redundant features
    n_classes=4,  # Number of classes
    random_state=42
)
data = Data(X, y, name="RandomData")
data.split_train_test(test_size=0.2, random_state=2)

opt_paras = {"name": "GA", "epoch": 30, "pop_size": 30}
model = MhaElmClassifier(layer_sizes=10, act_name="elu", obj_name="KLDL", optim="BaseGA",
                         optim_params=opt_paras, verbose=True, seed=42)
model.fit(data.X_train, data.y_train, lb=(-10, ), ub=(10, ))
y_pred = model.predict(data.X_test)

## Get parameters for model
print(model.get_params())

## Get weights of neural network (ELM network)
print(model.network.get_weights())
