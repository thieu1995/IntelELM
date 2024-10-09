#!/usr/bin/env python
# Created by "Thieu" at 22:28, 14/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from intelelm import get_dataset, MhaElmClassifier, Data
from sklearn.datasets import make_classification

# Introduce hyper-parameters
TEST_SIZE = 0.2
RANDOM_STATE = 2
LAYER_SIZES = (10, 5)
ACTIVATION = "relu"
OPTIMIZER = "BaseGA"
OPT_PARAMS = {"name": "GA", "epoch": 10, "pop_size": 30}
VERBOSE = True
SEED = 42


def build_and_evaluate_model(data, obj_name, evaluation_methods):
    model = MhaElmClassifier(layer_sizes=LAYER_SIZES,
                             act_name=ACTIVATION,
                             obj_name=obj_name,
                             optim=OPTIMIZER,
                             optim_paras=OPT_PARAMS,
                             verbose=VERBOSE,
                             seed=SEED)
    model.fit(data.X_train, data.y_train)
    predictions = model.predict(data.X_test)
    print(model.evaluate(data.y_test, predictions, list_metrics=evaluation_methods))
    print(model.get_weights())


def try_random_dataset():
    X, y = make_classification(
        n_samples=300,
        n_features=7,
        n_informative=3,
        n_redundant=0,
        n_classes=4,
        random_state=42
    )
    data = Data(X, y, name="RandomData")
    data.split_train_test(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    build_and_evaluate_model(data, "KLDL", ("AS", "PS", "F1S"))


def try_intelelm_binary_dataset():
    print(MhaElmClassifier.CLS_OBJ_LOSSES)
    data = get_dataset("circles")
    data.split_train_test(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(data.X_train.shape, data.X_test.shape)
    build_and_evaluate_model(data, "BSL", ("AS",))


def try_intelelm_multi_class_dataset():
    data = get_dataset("blobs")
    data.split_train_test(test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(data.X_train.shape, data.X_test.shape)
    build_and_evaluate_model(data, "KLDL", ("AS", "PS"))


try_random_dataset()
try_intelelm_binary_dataset()
try_intelelm_multi_class_dataset()
