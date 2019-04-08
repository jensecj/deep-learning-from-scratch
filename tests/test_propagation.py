import numpy as np
from numpy.testing import assert_allclose

import dlfs.activation_functions as A
from dlfs.propagation import *

def test_linear_forward():
    activations = np.array([[ 1.62434536, -0.61175641],
                            [-0.52817175, -1.07296862],
                            [ 0.86540763, -2.3015387 ]])
    weights = np.array([[ 1.74481176, -0.7612069, 0.3190391 ]])
    biases = np.array([[-0.24937038]])

    Z, linear_cache = linear_forward(activations, weights, biases)

    assert_allclose(Z, np.array([[3.26295337, -1.23429987]]), rtol=1e-5, atol=0)

def test_linear_activation_forward():
    prev_activations = np.array([[-0.41675785, -0.05626683],
                                 [-2.1361961,  1.64027081],
                                 [-1.79343559, -0.84174737]])

    weights = np.array([[ 0.50288142, -1.24528809, -1.05795222]])
    biases = np.array([[-0.90900761]])

    Z, cache = linear_activation_forward(prev_activations, weights, biases, A.sigmoid)
    assert_allclose(Z, np.array([[0.96890023, 0.11013289]]), rtol=1e-5, atol=0)

    Z, cache = linear_activation_forward(prev_activations, weights, biases, A.relu)
    assert_allclose(Z, np.array([[3.43896131, 0.]]), rtol=1e-5, atol=0)

def test_L_model_forward():
    # 5 unit input, 4 units out
    W1 = np.array([[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
                   [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
                   [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
                   [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]])

    b1 = np.array([[ 1.38503523],
                   [-0.51962709],
                   [-0.78015214],
                   [ 0.95560959]])

    # 4 unit input, 3 units out
    W2 = np.array([[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
                   [-0.56147088, -1.0335199,   0.35877096,  1.07368134],
                   [-0.37550472,  0.39636757, -0.47144628,  2.33660781]])

    b2 = np.array([[ 1.50278553],
                   [-0.59545972],
                   [ 0.52834106]])

    # 3 unit input, 1 unit out
    W3 = np.array([[0.9398248 ,  0.42628539, -0.75815703]])

    b3 = np.array([[-0.16236698]])

    prev_activations = np.array([[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                                 [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                                 [ 1.63929108, -0.4298936, 2.63128056, 0.60182225],
                                 [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                                 [0.07612761, -0.15512816, 0.63422534, 0.810655]])

    weights = {1: W1, 2: W2, 3: W3}
    biases = {1: b1, 2: b2, 3: b3}

    last_activations, cache = L_model_forward(prev_activations, weights, biases)
    assert_allclose(last_activations, np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]]))
    assert(len(cache) == 3)
