#!/usr/bin/env python
# Created by "Thieu" at 10:02, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def celu(x, alpha=1.0):
    return np.maximum(0, x) + np.minimum(0, alpha*(np.exp(x / alpha) - 1))


def prelu(x, alpha=0.5):
    return np.where(x < 0, alpha*x, x)


def gelu(x, alpha=0.044715):
    return x/2 * (1 + np.tanh(np.sqrt(2.0/np.pi) * (x + alpha*x**3)))


def elu(x, alpha=1):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)


def selu(x, alpha=1.67326324, scale=1.05070098):
    return np.where(x < 0, scale*alpha*(np.exp(x) - 1), scale*x)


def rrelu(x, lower=1./8, upper=1./3):
    alpha = np.random.uniform(lower, upper)
    return np.where(x < 0, alpha*x, x)


def tanh(x):
    return np.tanh(x)


def hard_tanh(x, lower=-1., upper=1.):
    return np.where(x < lower, -1, np.where(x > upper, upper, x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hard_sigmoid(x, lower=-2.5, upper=2.5):
    return np.where(x < lower, 0, np.where(x > upper, 1, 0.2*x + 0.5))


def log_sigmoid(x):
    return -np.log(1 + np.exp(-x))


def swish(x):
    # = silu (pytorch)
    return x / (1. + np.exp(-x))


def hard_swish(x, lower=-3., upper=3.):
    return np.where(x <= lower, 0, np.where(x >= upper, x, x*(x + 3)/6))


def soft_plus(x, beta=1.0):
    return 1.0/beta * np.log(1 + np.exp(beta * x))


def mish(x, beta=1.0):
    return x * np.tanh(1.0/beta * np.log(1 + np.exp(beta * x)))


def soft_sign(x):
    return x / (1 + np.abs(x))


def tanh_shrink(x):
    return x - np.tanh(x)


def soft_shrink(x, alpha=0.5):
    return np.where(x < -alpha, x + alpha, np.where(x > alpha, x - alpha, 0))


def hard_shrink(x, alpha=0.5):
    return np.where(-alpha < x < alpha, x, 0)


def softmin(x):
    exp_x = np.exp(-x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def log_softmax(x):
    log_exp_x = x - np.max(x, axis=-1, keepdims=True)
    log_exp_x = log_exp_x - np.log(np.sum(np.exp(log_exp_x), axis=-1, keepdims=True))
    return log_exp_x


silu = swish
