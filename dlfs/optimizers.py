import numpy as np

from dlfs.core import split_dataset

def mini_batch_gradient_descent(parameters, grads, learning_rate, batch_size):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    # this is the derivative of the cost function
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters


def stochastic_gradient_descent(parameters, grads, learning_rate):
    """computes gradients for examples, one at a time."""
    return mini_batch_gradient_descent(parameters, grads, learning_rate, 1)

def batch_gradient_descent(parameters, grads, learning_rate):
    return mini_batch_gradient_descent(parameters, grads, learning_rate, 1)

def RMSProp():
    pass

def Adam():
    pass
