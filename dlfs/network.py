import numpy as np

from dlfs.core import Layer, Network
import dlfs.propagation as P

def predict(net, X, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    # Forward propagation
    probas, caches = P.forward_propagation(net, X, parameters)

    # convert probas to 0/1 predictions
    p = np.round(probas)

    return p

def train(net, X, Y, learning_rate, num_iterations):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    net -- The networks structure, layers, cost function, optimizer, etc.
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector, of shape (1, number of examples)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    costs = [] # keep track of cost

    # Parameters initialization.
    parameters = P.initialize_parameters(net)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = P.forward_propagation(net, X, parameters)

        # Compute cost.
        cost = net.cost(AL, Y)

        # Backward propagation.
        grads = P.backward_propagation(net, AL, Y, caches)

        # Update parameters.
        parameters = net.optimizer(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)

    return parameters, costs
