#!/usr/bin/env python
# Created by "Thieu" at 22:28, 14/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from intelelm import get_dataset, MhaElmClassifier, Data
from sklearn.datasets import make_classification


def try_random_dataset():
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

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="KLDL", optimizer="BaseGA",
                             optimizer_paras=opt_paras, verbose=True, seed=42)
    model.fit(data.X_train, data.y_train)
    y_pred = model.predict(data.X_test)
    print(model.evaluate(data.y_test, y_pred, list_metrics=("AS", "PS", "F1S")))


def try_intelelm_binary_dataset():
    print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)

    data = get_dataset("circles")
    data.split_train_test(test_size=0.2, random_state=2)
    print(data.X_train.shape, data.X_test.shape)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="BSL", optimizer="BaseGA",
                             optimizer_paras=opt_paras, verbose=True, seed=42)
    model.fit(data.X_train, data.y_train)

    print(model.score(data.X_test, data.y_test, method="AS"))
    print(model.scores(data.X_test, data.y_test, list_methods=("AS", "PS")))


def try_intelelm_multi_class_dataset():
    data = get_dataset("blobs")
    data.split_train_test(test_size=0.2, random_state=2)
    print(data.X_train.shape, data.X_test.shape)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="KLDL", optimizer="BaseGA",
                             optimizer_paras=opt_paras, verbose=True, seed=42)
    model.fit(data.X_train, data.y_train)

    y_pred = model.predict(data.X_test)
    print(model.evaluate(data.y_test, y_pred, list_metrics=("AS", "PS")))


try_random_dataset()
try_intelelm_binary_dataset()
try_intelelm_multi_class_dataset()
