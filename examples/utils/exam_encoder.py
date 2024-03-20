#!/usr/bin/env python
# Created by "Thieu" at 23:29, 24/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import Data


X = np.array([[ 1., -2.,  2.],
                 [ -2.,  1.,  3.],
                 [ 4.,  1., -2.]])
y = np.array([[1, 2, 0],
              [0, 0, 1],
              [0, 2, 2]])

y = np.array([[1, 2, 0]])

data = Data(X, y)
y, le = data.encode_label(y)
print(y)
