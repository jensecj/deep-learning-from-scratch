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
    for l in range(L-1):
        prev_activations = activations
        activations, cache = linear_activation_forward(prev_activations, weights[l], biases[l], A.relu)
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    last_activations, cache = linear_activation_forward(activations, weights[L-1], biases[L-1], A.sigmoid)
    caches.append(cache)

    assert(last_activations.shape == (1, X.shape[1]))

    return last_activations, caches

def linear_backward(gradient, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    gradient -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (prev_activations, weights, biases) coming from the forward propagation in the current layer

    Returns:
    prev_activation_gradient -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as prev_activations
    weight_gradient -- Gradient of the cost with respect to weights (current layer l), same shape as weights
    bias_gradient -- Gradient of the cost with respect to biases (current layer l), same shape as biases
    """
    prev_activations, weights, biases = cache
    m = prev_activations.shape[1]

    weight_gradient = 1/m * np.dot(gradient, prev_activations.T)
    bias_gradient = 1/m * np.sum(gradient, axis=1, keepdims=True)
    prev_activation_gradient = np.dot(weights.T, gradient)

    assert (prev_activation_gradient.shape == prev_activations.shape)
    assert (weight_gradient.shape == weights.shape)
    assert (bias_gradient.shape == biases.shape)

    return prev_activation_gradient, weight_gradient, bias_gradient

def linear_activation_backward(activation_gradient, cache, activation_function):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    activation_gradient -- post-activation_function gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation_function -- the activation_function to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    prev_activation_gradient -- Gradient of the cost with respect to the activation_function (of the previous layer l-1), same shape as A_prev
    weight_gradient -- Gradient of the cost with respect to W (current layer l), same shape as W
    bias_gradient -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    gradient = activation_function(activation_gradient, activation_cache)
    prev_activation_gradient, weight_gradient, bias_gradient = linear_backward(gradient, linear_cache)

    return prev_activation_gradient, weight_gradient, bias_gradient

def L_model_backward(last_activations, labels, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    last_activations -- probability vector, output of the forward propagation (L_model_forward())
    labels -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    gradients -- A dictionary with the gradients
             gradients["dA" + str(l)] = ...
             gradients["dW" + str(l)] = ...
             gradients["db" + str(l)] = ...
    """
    L = len(caches) # the number of layers
    m = last_activations.shape[1]
    labels = labels.reshape(last_activations.shape) # after this line, labels is the same shape as last_activations

    # We start from the last layer and work backwards.
    # start with the last cache.
    current_cache = caches[L-1]

    # the derivative of the cost with respect to the last activations
    last_activation_derivative = - (np.divide(labels, last_activations) - np.divide(1 - labels, 1 - last_activations))

    activation_gradients = {}
    weight_gradients = {}
    bias_gradients = {}

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "last_activation_derivative, current_cache". Outputs: "gradients["last_activation_derivative-1"], gradients["dWL"], gradients["dbL"]
    dA, dW, db = linear_activation_backward(last_activation_derivative, current_cache, A.sigmoid_deriv)
    activation_gradients[L-1] = dA
    weight_gradients[L] = dW
    bias_gradients[L] = db

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "gradients["dA" + str(l + 1)], current_cache". Outputs: "gradients["dA" + str(l)] , gradients["dW" + str(l + 1)] , gradients["db" + str(l + 1)]
        current_cache = caches[l]

        dA, dW, db = linear_activation_backward(activation_gradients[l + 1], current_cache, A.relu_deriv)
        activation_gradients[l] = dA
        weight_gradients[l + 1] = dW
        bias_gradients[l + 1] = db

    return activation_gradients, weight_gradients, bias_gradients
