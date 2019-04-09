import numpy as np

def batch_gradient_descent(weights, biases, weight_gradients, bias_gradients, learning_rate = 0.01):
    """compute gradients for all examples at once."""
    new_weights = [None] * len(weights)
    new_biases = [None] * len(biases)

    # Update rule for each parameter. Use a for loop.
    for l in range(len(biases)):
        new_weights[l] = weights[l] - (learning_rate * weight_gradients[l+1])
        new_biases[l] = biases[l] - (learning_rate * bias_gradients[l+1])

    return new_weights, new_biases
