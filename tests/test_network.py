import numpy as np
from numpy.testing import assert_allclose

import dlfs.activation_functions as A
import dlfs.cost_functions as C
import dlfs.optimization_functions as O
import dlfs.network as nn
from dlfs.network import Layer, Network

def test_initialize_network_with_one_hidden_layer():
    layers = [ Layer(5, A.relu), Layer(4, A.relu), Layer(1, A.sigmoid) ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    weights, biases = nn.initialize(net)

    print(weights)
    assert(weights[0].shape == (4, 5))
    assert(weights[1].shape == (1, 4))

    print(biases)
    assert(biases[0].shape == (4,1))
    assert(biases[1].shape == (1,1))

def test_initialize_network_with_many_hidden_layer():
    layers = [ Layer(10, A.relu), Layer(6, A.relu), Layer(6, A.relu), Layer(3, A.relu), Layer(2, A.sigmoid) ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    weights, biases = nn.initialize(net)

    print(weights)
    assert(weights[0].shape == (6, 10))
    assert(weights[1].shape == (6, 6))
    assert(weights[2].shape == (3, 6))
    assert(weights[3].shape == (2, 3))

    print(biases)
    assert(biases[0].shape == (6,1))
    assert(biases[1].shape == (6,1))
    assert(biases[2].shape == (3,1))
    assert(biases[3].shape == (2,1))

def test_predict():
    W1 = np.array([[-0.00615039,  0.0169021 ],
                   [-0.02311792,  0.03137121],
                   [-0.0169217 , -0.01752545],
                   [ 0.00935436, -0.05018221]])

    W2 = np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]])

    b1 = np.array([[-8.97523455e-07],
                   [ 8.15562092e-06],
                   [ 6.04810633e-07],
                   [-2.54560700e-06]])

    b2 = np.array([[ 9.14954378e-05]])

    weights = [W1, W2]
    biases = [b1, b2]

    np.random.seed(1)
    X = np.random.randn(2, 3)

    input_layers = X.shape[0]
    output_layers = 1

    layers = [ Layer(input_layers, A.relu), Layer(4, A.relu), Layer(output_layers, A.sigmoid) ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    predictions = nn.predict(net, X, weights, biases)
    predictions = np.round(predictions)

    assert_allclose(np.mean(predictions), 0.666666667)

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m / 2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m, D)) # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j, N * (j + 1))
        t = np.linspace(j*3.12, (j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

def test_network():
    inputs, labels = load_planar_dataset()

    print(inputs.shape)
    print(labels.shape)

    input_layers = inputs.shape[0]
    output_layers = 1

    layers = [ Layer(input_layers, A.relu),
               Layer(10, A.relu),
               Layer(5, A.relu),
               Layer(output_layers, A.sigmoid) ]

    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    weights, biases = nn.initialize(net)
    weights, biases, costs = nn.train(net, inputs, labels, weights, biases, 10000, 0.1)
    predictions = nn.predict(net, inputs, weights, biases)

    accuracy = float((np.dot(labels, predictions.T) + np.dot(1-labels, 1-predictions.T)) / float(labels.size)*100)
    print(f"accuracy: {accuracy}")

    assert({} == {1})
