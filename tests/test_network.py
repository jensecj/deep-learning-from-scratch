import numpy as np
from numpy.testing import assert_allclose

import dlfs.activations as A
import dlfs.cost_functions as C
import dlfs.optimizers as O
import dlfs.network as nn
from dlfs.network import Layer, Network

def test_initialize_network_with_one_hidden_layer():
    layers = [ Layer(5),
               Layer(4, A.relu),
               Layer(1, A.sigmoid) ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    parameters = nn._initialize_parameters(net)

    print(parameters)
    assert(parameters["W1"].shape == (4, 5))
    assert(parameters["W2"].shape == (1, 4))
    assert(parameters["b1"].shape == (4,1))
    assert(parameters["b2"].shape == (1,1))

def test_initialize_network_with_many_hidden_layers():
    layers = [ Layer(10),
               Layer(6, A.relu),
               Layer(6, A.relu),
               Layer(3, A.relu),
               Layer(2, A.sigmoid) ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    parameters = nn._initialize_parameters(net)

    print(parameters)

    assert(parameters["W1"].shape == (6, 10))
    assert(parameters["W2"].shape == (6, 6))
    assert(parameters["W3"].shape == (3, 6))
    assert(parameters["W4"].shape == (2, 3))

    assert(parameters["b1"].shape == (6,1))
    assert(parameters["b2"].shape == (6,1))
    assert(parameters["b3"].shape == (3,1))
    assert(parameters["b4"].shape == (2,1))

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

    input_layers = X.shape[0]
    output_layers = 1

    layers = [
        Layer(input_layers),
        Layer(4, A.tanh),
        Layer(output_layers, A.sigmoid)
    ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    predictions = nn.predict(net, X, parameters)
    print(predictions)
    predictions = np.round(predictions)

    # assert(predictions.shape == (input_layers, 1))

    # assert_allclose(np.mean(predictions), 0.666666667)

###############
# propagation #
###############

def test_linear_forward():
    activations = np.array([[ 1.62434536, -0.61175641],
                            [-0.52817175, -1.07296862],
                            [ 0.86540763, -2.3015387 ]])
    weights = np.array([[ 1.74481176, -0.7612069, 0.3190391 ]])
    biases = np.array([[-0.24937038]])

    Z, linear_cache = nn._linear_forward(activations, weights, biases)

    assert_allclose(Z, np.array([[3.26295337, -1.23429987]]), rtol=1e-5, atol=0)

def test_linear_activation_forward():
    prev_activations = np.array([[-0.41675785, -0.05626683],
                                 [-2.1361961,  1.64027081],
                                 [-1.79343559, -0.84174737]])

    weights = np.array([[ 0.50288142, -1.24528809, -1.05795222]])
    biases = np.array([[-0.90900761]])

    Z, cache = nn._linear_activation_forward(prev_activations, weights, biases, A.sigmoid)
    assert_allclose(Z, np.array([[0.96890023, 0.11013289]]), rtol=1e-5, atol=0)

    Z, cache = nn._linear_activation_forward(prev_activations, weights, biases, A.relu)
    assert_allclose(Z, np.array([[3.43896131, 0.]]), rtol=1e-5, atol=0)

def test_forward_propagation():
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

    parameters = { "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3 }

    layers = [ Layer(5),
               Layer(5, A.relu),
               Layer(4, A.relu),
               Layer(1, A.sigmoid) ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    last_activations, cache = nn._forward_propagation(net, prev_activations, parameters)
    assert_allclose(last_activations, np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]]), rtol=1e-5, atol=0)
    assert(len(cache) == 3)

def test_linear_backward():
    dZ = np.array([[ 1.62434536, -0.61175641]])

    A = np.array([[-0.52817175, -1.07296862],
                  [ 0.86540763, -2.3015387 ],
                  [ 1.74481176, -0.7612069 ]])

    W = np.array([[ 0.3190391 , -0.24937038, 1.46210794]])

    b = np.array([[-2.06014071]])

    linear_cache = (A, W, b)

    prev_activation_derivatives, weight_derivatives, bias_derivatives = nn._linear_backward(dZ, linear_cache)

    assert(len(prev_activation_derivatives) == 3)

    assert_allclose(prev_activation_derivatives, np.array([[ 0.51822968, -0.19517421],
                                                           [-0.40506361, 0.15255393],
                                                           [ 2.37496825, -0.89445391]]), rtol=1e-6, atol=0)
    assert_allclose(weight_derivatives, np.array([[-0.10076895, 1.40685096, 1.64992505]]), rtol=1e-6, atol=0)
    assert_allclose(bias_derivatives, np.array([[ 0.50629448]]), rtol=1e-6, atol=0)

def test_linear_activation_backward():
    last_activations_derivative = np.array([[-0.41675785, -0.05626683]])
    W = np.array([[-1.05795222, -0.90900761,  0.55145404]])
    b = np.array([[2.29220801]])
    activations = np.array([[-2.1361961 ,  1.64027081],
                            [-1.79343559, -0.84174737],
                            [ 0.50288142, -1.24528809]])

    linear_cache = (activations, W, b)
    activation_cache = np.array([[0.04153939, -1.11792545]])

    linear_activation_cache = (linear_cache, activation_cache)

    prev_activation_derivatives, weight_derivatives, bias_derivatives = nn._linear_activation_backward(last_activations_derivative, linear_activation_cache, A.sigmoid)
    assert_allclose(prev_activation_derivatives, np.array([[ 0.11017994,  0.01105339],
                                                           [ 0.09466817,  0.00949723],
                                                           [-0.05743092, -0.00576154]]), rtol=1e-5, atol=0)
    assert_allclose(weight_derivatives, np.array([[ 0.10266786, 0.09778551, -0.01968084]]), rtol=1e-5, atol=0)
    assert_allclose(bias_derivatives, np.array([[-0.05729622]]), rtol=1e-5, atol=0)

    prev_activation_derivatives, weight_derivatives, bias_derivatives = nn._linear_activation_backward(last_activations_derivative, linear_activation_cache, A.relu)
    assert_allclose(prev_activation_derivatives, np.array([[ 0.44090989, 0. ],
                                                           [ 0.37883606, 0. ],
                                                           [-0.2298228, 0. ]]), rtol=1e-5, atol=0)
    assert_allclose(weight_derivatives, np.array([[ 0.44513824, 0.37371418, -0.10478989]]), rtol=1e-5, atol=0)
    assert_allclose(bias_derivatives, np.array([[-0.20837892]]), rtol=1e-5, atol=0)

def test_backward_propagation():
    np.random.seed(3)
    last_activations = np.array([[1.78862847, 0.43650985]])
    labels = np.array([[1, 0]])

    layer_1_activations = np.array([[ 0.09649747, -1.8634927 ],
                                    [-0.2773882 , -0.35475898],
                                    [-0.08274148, -0.62700068],
                                    [-0.04381817, -0.47721803]])
    layer_1_weights = np.array([[-1.31386475,   0.88462238,  0.88131804,  1.70957306],
                                [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
                                [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]])
    layer_1_biases = np.array([[ 1.48614836],
                               [ 0.23671627],
                               [-1.02378514]])
    layer_1_linear_output = np.array([[-0.7129932,   0.62524497],
                                      [-0.16051336, -0.76883635],
                                      [-0.23003072,  0.74505627]])
    linear_cache_activation_1 = ((layer_1_activations, layer_1_weights, layer_1_biases), layer_1_linear_output)

    layer_2_activations = np.array([[ 1.97611078, -1.24412333],
                                    [-0.62641691, -0.80376609],
                                    [-2.41908317, -0.92379202]])
    layer_2_weights = np.array([[-1.02387576,  1.12397796, -0.13191423]])
    layer_2_biases = np.array([[-1.62328545]])
    layer_2_linear_output = np.array([[ 0.64667545, -0.35627076]])

    linear_cache_activation_2 = ((layer_2_activations, layer_2_weights, layer_2_biases), layer_2_linear_output)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    layers = [ Layer(3),
               Layer(2, A.relu),
               Layer(1, A.sigmoid) ]
    net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

    # activation_gradients, weight_gradients, bias_gradients = nn._backward_propagation(net, last_activations, labels, caches)

    # assert_allclose(activation_gradients[1], np.array([[ 0.12913162, -0.44014127],
    #                                                    [-0.14175655, 0.48317296],
    #                                                    [ 0.01663708, -0.05670698]]), rtol=1e-5, atol=0)

    # assert_allclose(weight_gradients[1], np.array([[ 0.41010002, 0.07807203, 0.13798444, 0.10502167],
    #                                                [ 0., 0., 0., 0. ],
    #                                                [ 0.05283652, 0.01005865, 0.01777766, 0.0135308 ]]), rtol=1e-5, atol=0)

    # assert_allclose(bias_gradients[1], np.array([[-0.22007063],
    #                                              [ 0. ],
    #                                              [-0.02835349]]), rtol=1e-5, atol=0)
