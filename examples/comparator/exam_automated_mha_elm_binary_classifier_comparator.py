#!/usr/bin/env python
# Created by "Thieu" at 23:30, 17/08/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from intelelm import get_dataset, AutomatedMhaElmComparator


data = get_dataset("circles")
data.split_train_test(test_size=0.2, random_state=2)
print(data.X_train.shape, data.X_test.shape)

data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=('minmax', ))
data.X_test = scaler_X.transform(data.X_test)

# Example optimizer dict
optimizer_dict = {
    'BaseGA':       {"epoch": 10, "pop_size": 20},
    "OriginalPSO":  {"epoch": 10, "pop_size": 20},
}

# Initialize the comparator
comparator = AutomatedMhaElmComparator(
    optimizer_dict=optimizer_dict,
    task="classification",
    layer_sizes=(10, ),
    act_name="elu",
    obj_name="F1S",
    verbose=False,
    seed=42,
)

# Perform comparison
# results = comparator.compare_cross_val_score(data.X_train, data.y_train, metric="AS", cv=4, n_trials=2, to_csv=True)
# print(results)

# results = comparator.compare_cross_validate(data.X_train, data.y_train, metrics=["AS", "PS", "F1S", "NPV"],
#                                            cv=4, return_train_score=True, n_trials=2, to_csv=True)
# print(results)

results = comparator.compare_train_test(data.X_train, data.y_train, data.X_test, data.y_test,
                                        metrics=["AS", "PS", "F1S", "NPV"], n_trials=2, to_csv=True)
print(results)
