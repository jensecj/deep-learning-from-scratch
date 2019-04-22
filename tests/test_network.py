import numpy as np
from numpy.testing import assert_allclose

import dlfs.activations as A
import dlfs.cost_functions as C
import dlfs.optimizers as O
import dlfs.network as N
import dlfs.propagation as P
from dlfs.network import Layer, Network

# TODO: test no layers, one layer, two layers

def test_initialize_network_with_one_hidden_layer():
    layers = [ Layer(5),
               Layer(4, A.relu),
               Layer(1, A.sigmoid) ]
    net = Network(layers, None, None)

    parameters = P.initialize_parameters(net)

    print(parameters)

    assert parameters["W1"].shape == (4, 5)
    assert parameters["W2"].shape == (1, 4)
    assert parameters["b1"].shape == (4,1)
    assert parameters["b2"].shape == (1,1)

def test_initialize_network_with_many_hidden_layers():
    layers = [ Layer(10),
               Layer(6, A.relu),
               Layer(6, A.relu),
               Layer(3, A.relu),
               Layer(2, A.sigmoid) ]
    net = Network(layers, None, None)

    parameters = P.initialize_parameters(net)

    print(parameters)

    assert parameters["W1"].shape == (6, 10)
    assert parameters["W2"].shape == (6, 6)
    assert parameters["W3"].shape == (3, 6)
    assert parameters["W4"].shape == (2, 3)

    assert parameters["b1"].shape == (6,1)
    assert parameters["b2"].shape == (6,1)
    assert parameters["b3"].shape == (3,1)
    assert parameters["b4"].shape == (2,1)

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

    parameters = { "W1": W1, "b1": b1, "W2": W2, "b2": b2 }

    np.random.seed(1)
    X = np.random.randn(2, 3)

    input_layers = X.shape[1]
    output_layers = 1

    layers = [
        Layer(input_layers),
        Layer(4, A.tanh),
        Layer(output_layers, A.sigmoid)
    ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    predictions = N.predict(net, X, parameters)
    predictions = np.round(predictions)
    print(predictions)

    assert predictions.shape == (1, input_layers)
    assert_allclose(np.mean(predictions), 0.666666667)

def test_train():
    pass
    # assert False
