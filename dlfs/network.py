from dataclasses import dataclass
from typing import Any, List, Callable, Tuple

import dlfs.propagation as P
import dlfs.cost_functions as C

import numpy as np

# TODO: specify Callables once numpy gets proper typing stubs

@dataclass(frozen=True)
class Layer:
    hidden_units: int
    activation: Any

# TODO: fns for dimensions / num_layers?
@dataclass
class Network:
    layers: List[Layer]
    cost: Any
    optimizer: Any

def initialize(net: Network):
    network_dimensions = [ l.hidden_units for l in net.layers ]
    num_layers = len(network_dimensions)

    print("network dimensions: {0}".format(network_dimensions))
    print("network has {0} layers".format(num_layers))

    weights = [None] * (num_layers-1)
    biases = [None] * (num_layers-1)

    for l in range(1, num_layers):
        weights[l-1] = np.random.randn(network_dimensions[l], network_dimensions[l-1]) / np.sqrt(network_dimensions[l-1]) * 0.01
        biases[l-1] = np.zeros((network_dimensions[l], 1))

    return weights, biases

def train(net: Network, inputs, labels, weights, biases, iterations:int = 1000, learning_rate:int = 0.01):
    costs = []

    for i in range(0, iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        last_activations, caches = P.L_model_forward(inputs, weights, biases)

        # Compute cost.
        cost = net.cost(last_activations, labels)

        # Backward propagation.
        activation_gradients, weight_gradients, bias_gradients = P.L_model_backward(last_activations, labels, caches)

        # Update parameters.
        weights, biases = net.optimizer(weights, biases, weight_gradients, bias_gradients, learning_rate)

        # Print the cost every 100 training example
        if i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    return weights, biases, costs


def predict(net:Network, X, weights, biases):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(biases) # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    predictions, caches = P.L_model_forward(X, weights, biases)

    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    # print("Accuracy: "  + str(np.sum((p == y)/m)))

    return predictions
