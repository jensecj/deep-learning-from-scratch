import numpy as np
import dlfs.activation_functions as A

def linear_forward(activations, weights, biases):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    activations -- activations from previous layer (or input data): (size of previous layer, number of examples)
    weights -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    biases -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "activations", "weights" and "biases" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(weights, activations) + biases

    assert(Z.shape == (weights.shape[0], activations.shape[1]))

    linear_cache = (activations, weights, biases)
    return Z, linear_cache

def linear_activation_forward(prev_activation, weights, biases, activation_function):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    prev_activation -- activations from previous layer (or input data): (size of previous layer, number of examples)
    weights -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    biases -- bias vector, numpy array of shape (size of the current layer, 1)
    activation_function -- the activation_function to be used in this layer

    Returns:
    A -- the output of the activation_function function, also called the post-activation_function value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    # Inputs: "prev_activation, weights, biases". Outputs: "A, activation_cache".
    Z, linear_cache = linear_forward(prev_activation, weights, biases)
    A, activation_cache = activation_function(Z)

    assert (A.shape == (weights.shape[0], prev_activation.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, weights, biases):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    last_activations -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    caches = []
    activations = X
    L = len(biases) # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        print('in layer {0} pa: {1}, w: {2}, b: {3}'.format(l, activations.shape, weights[l].shape, biases[l].shape))
        prev_activations = activations
        activations, cache = linear_activation_forward(prev_activations, weights[l], biases[l], A.relu)
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    last_activations, cache = linear_activation_forward(activations, weights[L], biases[L], A.sigmoid)
    caches.append(cache)

    assert(last_activations.shape == (1, X.shape[1]))

    return last_activations, caches
