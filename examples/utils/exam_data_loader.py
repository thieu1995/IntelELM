#!/usr/bin/env python
# Created by "Thieu" at 22:28, 14/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from intelelm import get_dataset

# Try unknown data
get_dataset("unknown")
# Enter: 1

data = get_dataset("Arrhythmia")
data.split_train_test(test_size=0.2)

print(data.X_train[:2].shape)
print(data.y_train[:2].shape)
