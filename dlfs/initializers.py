# https://towardsdatascience.com/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e?gi=412f60fca20b
# https://machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/
# https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks
# https://cs231n.github.io/neural-networks-2/#init

# Xavier initialization: http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
# He initialization: https://arxiv.org/pdf/1502.01852v1.pdf

import numpy as np

def zeros(shape):
    return np.zeros(shape)

def ones(shape):
    return np.ones(shape)

def constant(shape, c):
    n = zeros(shape)
    n.fill(c)
    return n

def random(shape):
    return np.random.randn(shape[0], shape[1])

def variance_scaled(shape, fan_in):
    return random(shape) * (1/np.sqrt(fan_in))

def xavier(shape, fan_in, fan_out):
    return random(shape) * np.sqrt(2/(fan_in + fan_out))

def xavier_normal(shape, fan_in, fan_out):
    return random(shape) * (np.sqrt(6) / np.sqrt(fan_in + fan_out))

def he(shape, fan_in):
    return random(shape) * np.sqrt(2/fan_in)
