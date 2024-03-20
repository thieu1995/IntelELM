#!/usr/bin/env python
# Created by "Thieu" at 12:49, 17/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from intelelm import DataTransformer, Data


X = np.array([[ 1., -2.,  2.],
                 [ -2.,  1.,  3.],
                 [ 4.,  1., -2.]])

## Get all supported scaling methods
print(DataTransformer.SUPPORTED_SCALERS.keys())

### 1) Using only standard scaler
scaler = DataTransformer(scaling_methods="standard")
X_scaled = scaler.fit_transform(X)
X_unscaled = scaler.inverse_transform(X_scaled)

# Print the results
print("Original Data:")
print(X)
print("Scaled Data:")
print(X_scaled)
print("Transformed Back to Original:")
print(X_unscaled)


### 2) Using multiple  scalers
scaler = DataTransformer(scaling_methods=("standard", "minmax"))                # Just like Pipeline
X_scaled = scaler.fit_transform(X)
X_unscaled = scaler.inverse_transform(X_scaled)

# Print the results
print("\nOriginal Data:")
print(X)
print("Scaled Data:")
print(X_scaled)
print("Transformed Back to Original:")
print(X_unscaled)


### 3) Use methods in Data class instead
data = Data(X)
X_scaled, scaler = data.scale(X, scaling_methods=("standard", "minmax"))        # Just like Pipeline
X_unscaled = scaler.inverse_transform(X_scaled)

# Print the results
print("\nOriginal Data:")
print(X)
print("Scaled Data:")
print(X_scaled)
print("Transformed Back to Original:")
print(X_unscaled)


### 4) Use methods in Data class with parameters
data = Data(X)
X_scaled, scaler = data.scale(X, scaling_methods=("sinh-arc-sinh", "minmax"),
                              list_dict_paras=({"epsilon": 0.5, "delta": 2.5}, None))
X_unscaled = scaler.inverse_transform(X_scaled)

# Print the results
print("\nOriginal Data:")
print(X)
print("Scaled Data:")
print(X_scaled)
print("Transformed Back to Original:")
print(X_unscaled)


### 5) Use methods in Data class with parameters
data = Data(X)
X_scaled, scaler = data.scale(X, scaling_methods=("yeo-johnson", "sinh-arc-sinh"),
                              list_dict_paras=({"lmbda": 1.2}, {"epsilon": 0.5, "delta": 2.5}))
X_unscaled = scaler.inverse_transform(X_scaled)

# Print the results
print("\nOriginal Data:")
print(X)
print("Scaled Data:")
print(X_scaled)
print("Transformed Back to Original:")
print(X_unscaled)
