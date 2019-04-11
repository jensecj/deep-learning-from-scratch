import numpy as np
from numpy.testing import assert_allclose

from dlfs.network import Layer, Network
import dlfs.network as nn
import dlfs.activations as A
import dlfs.cost_functions as C
import dlfs.optimizers as O

def test_batch_gradient_descent():
    W1 = np.array([[-0.41675785, -0.05626683, -2.1361961,  1.64027081],
                   [-1.79343559, -0.84174737,  0.50288142, -1.24528809],
                   [-1.05795222, -0.90900761,  0.55145404,  2.29220801]])

    b1 = np.array([[ 0.04153939],
                   [-1.11792545],
                   [ 0.53905832]])

    W2 = np.array([[-0.5961597, -0.0191305,  1.17500122]])
    b2 = np.array([[-0.74787095]])

    weights = [W1, W2]
    biases = [b1, b2]

    dW1 = np.array([[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
                    [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
                    [-0.04381817, -0.47721803, -1.31386475,  0.88462238]])

    db1 = np.array([[0.88131804],
                    [1.70957306],
                    [0.05003364]])

    dW2 = np.array([[-0.40467741, -0.54535995, -1.54647732]])
    db2 = np.array([[0.98236743]])

    # FIXME: use lists for gradients
    weight_gradients = [0,dW1, dW2]
    bias_gradients = [0,db1, db2]

    weights, biases = O.batch_gradient_descent(weights, biases, weight_gradients, bias_gradients, 0.1)

    assert_allclose(weights[0], np.array([[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
                                          [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
                                          [-1.0535704, -0.86128581, 0.68284052, 2.20374577]]), rtol=1e-5, atol=0)

    assert_allclose(biases[0], np.array([[-0.04659241],
                                         [-1.28888275],
                                         [ 0.53405496]]), rtol=1e-5, atol=0)

    assert_allclose(weights[1], np.array([[-0.55569196, 0.0354055, 1.32964895]]), rtol=1e-5, atol=0)
    assert_allclose(biases[1], np.array([[-0.84610769]]), rtol=1e-5, atol=0)
