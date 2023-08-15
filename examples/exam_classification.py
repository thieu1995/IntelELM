#!/usr/bin/env python
# Created by "Thieu" at 22:28, 14/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from intelelm import get_dataset, MhaElmClassifier


def trial_binary_dataset():
    data = get_dataset("banknote")
    data.split_train_test(test_size=0.2, random_state=2)
    print(data.X_train.shape, data.X_test.shape)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)
    model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="BSL", optimizer="BaseGA", optimizer_paras=opt_paras)
    model.fit(data.X_train, data.y_train)

    pred = model.predict(data.X_test)
    print(pred)


def trial_multi_class_dataset():
    data = get_dataset("aniso")
    data.split_train_test(test_size=0.2, random_state=2)
    print(data.X_train.shape, data.X_test.shape)

    opt_paras = {"name": "GA", "epoch": 10, "pop_size": 30}
    print(MhaElmClassifier.SUPPORTED_CLS_OBJECTIVES)
    model = MhaElmClassifier(hidden_size=10, act_name="elu", obj_name="KLDL", optimizer="BaseGA", optimizer_paras=opt_paras)
    model.fit(data.X_train, data.y_train)

    pred = model.predict(data.X_test)
    print(pred)


trial_binary_dataset()
trial_multi_class_dataset()
