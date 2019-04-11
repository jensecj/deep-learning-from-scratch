import numpy as np
from numpy.testing import assert_allclose

import dlfs.activation_functions as A

###########
# sigmoid #
###########

def test_sigmoid_forward_with_negative():
    input = np.array([-1])
    activation, activation_cache = A.sigmoid.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))

    assert_allclose(activation, np.array([0.26894142]))

def test_sigmoid_forward_with_positive():
    input = np.array([1])
    activation, activation_cache = A.sigmoid.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([0.7310585786]))

def test_sigmoid_forward_with_zero():
    input = np.array([0])

    activation, activation_cache = A.sigmoid.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([0.5]))

def test_sigmoid_forward_with_row_vectors():
    input = np.array([1, 2, 3])

    activation, activation_cache = A.sigmoid.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([0.73105858, 0.88079708, 0.95257413]))

def test_sigmoid_forward_with_column_vector():
    input = np.array([[1.283462],
                      [0.023864],
                      [2.00981624]])

    activation, activation_cache = A.sigmoid.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([[0.78303851],
                                          [0.50596572],
                                          [0.8818238738]]))

def test_sigmoid_backward_with_positive_dA_positive_Z():
    dA = np.array([1.92])
    Z = np.array([1.045])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0.3695797134]))


def test_sigmoid_backward_with_negative_dA_positive_Z():
    dA = np.array([-1.32])
    Z = np.array([2.74])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([-0.07520700967]))

def test_sigmoid_backward_with_positive_dA_negative_Z():
    dA = np.array([0.2863])
    Z = np.array([-3.234])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0.01044113992]))

def test_sigmoid_backward_with_negative_dA_negative_Z():
    dA = np.array([-0.78234])
    Z = np.array([-3.9781])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([-0.01411295607]))

def test_sigmoid_backward_with_zero_dA_positive_Z():
    dA = np.array([0])
    Z = np.array([1.764832])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_sigmoid_backward_with_zero_dA_negative_Z():
    dA = np.array([0])
    Z = np.array([-1.764832])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_sigmoid_backward_with_positive_dA_zero_Z():
    dA = np.array([1.238645])
    Z = np.array([0])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0.30966125]))

def test_sigmoid_backward_with_negative_dA_zero_Z():
    dA = np.array([-0.973264])
    Z = np.array([0])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([-0.243316]))

def test_sigmoid_backward_with_zero_dA_zero_Z():
    dA = np.array([0])
    Z = np.array([0])

    dZ = A.sigmoid.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))


########
# tanh #
########

def test_tanh_forward_with_negative():
    input = np.array([-0.48])
    activation, activation_cache = A.tanh.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))

    assert_allclose(activation, np.array([-0.4462436]))

def test_tanh_forward_with_positive():
    input = np.array([1.39])
    activation, activation_cache = A.tanh.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))

    assert_allclose(activation, np.array([0.8831709]))

def test_tanh_forward_with_zero():
    input = np.array([0])
    activation, activation_cache = A.tanh.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))

    assert_allclose(activation, np.array([0]))

def test_tanh_forward_with_row_vector():
    input = np.array([2.26584, 0.89732, -0.7324])
    activation, activation_cache = A.tanh.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))

    assert_allclose(activation, np.array([0.978704, 0.7149904, -0.6245314]))

def test_tanh_forward_with_column_vector():
    input = np.array([[2.26584],
                      [0.89732],
                      [-0.7324]])
    activation, activation_cache = A.tanh.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))

    assert_allclose(activation, np.array([[0.978704],
                                          [0.7149904],
                                          [-0.6245314]]))


def test_tanh_backward_with_positive_dA_positive_Z():
    dA = np.array([1.324])
    Z = np.array([0.9264])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0.6205438]))

def test_tanh_backward_with_positive_dA_negative_Z():
    dA = np.array([0.7832])
    Z = np.array([-1.7234])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0.09370775]))

def test_tanh_backward_with_negative_dA_positive_Z():
    dA = np.array([-0.3423])
    Z = np.array([0.7268347])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([-0.21024192]))

def test_tanh_backward_with_negative_dA_negative_Z():
    dA = np.array([-1.69235])
    Z = np.array([-1.32])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([-0.42086311]))

def test_tanh_backward_with_zero_dA_positive_Z():
    dA = np.array([0])
    Z = np.array([1.32])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_tanh_backward_with_zero_dA_negative_Z():
    dA = np.array([0])
    Z = np.array([-5.6285])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_tanh_backward_with_positive_dA_zero_Z():
    dA = np.array([0.892164])
    Z = np.array([0])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0.892164]))

def test_tanh_backward_with_negative_dA_zero_Z():
    dA = np.array([-0.2695853])
    Z = np.array([0])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([-0.2695853]))

def test_tanh_backward_with_zero_dA_zero_Z():
    dA = np.array([0])
    Z = np.array([0])

    dZ = A.tanh.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))


########
# relu #
########

def test_relu_forward_with_negative():
    input = np.array([-1])

    activation, activation_cache = A.relu.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([0]))

def test_relu_forward_with_positive():
    input = np.array([1])

    activation, activation_cache = A.relu.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([1]))

def test_relu_forward_with_zero():
    input = np.array([0])

    activation, activation_cache = A.relu.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([0]))

def test_relu_forward_with_row_vector():
    input = np.array([-5, 2, -2, 11])

    activation, activation_cache = A.relu.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([0, 2, 0, 11]))

def test_relu_forward_with_column_vector():
    input = np.array([[1.283462],
                      [0.023864],
                      [2.00981624]])

    activation, activation_cache = A.relu.forward(input)

    assert(activation.shape == input.shape)
    assert(len(activation_cache) == len(input))
    assert(len(activation) == len(input))
    assert_allclose(activation, np.array([[1.283462],
                                          [0.023864],
                                          [2.00981624]]))

def test_relu_backward_with_positive_dA_positive_Z():
    dA = np.array([0.9732684])
    Z = np.array([1.764832])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0.9732684]))

def test_relu_backward_with_positive_dA_negative_Z():
    dA = np.array([0.9732684])
    Z = np.array([-1.764832])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_relu_backward_with_negative_dA_positive_Z():
    dA = np.array([-2.3765840])
    Z = np.array([0.47856])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([-2.3765840]))

def test_relu_backward_with_negative_dA_negative_Z():
    dA = np.array([-2.9655432])
    Z = np.array([-0.896349])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_relu_backward_with_zero_dA_positive_Z():
    dA = np.array([0])
    Z = np.array([0.454359])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_relu_backward_with_zero_dA_negative_Z():
    dA = np.array([0])
    Z = np.array([-1.34234])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_relu_backward_with_positive_dA_zero_Z():
    dA = np.array([0.78562])
    Z = np.array([0])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_relu_backward_with_negative_dA_zero_Z():
    dA = np.array([-1.57862])
    Z = np.array([0])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))

def test_relu_backward_with_zero_dA_zero_Z():
    dA = np.array([0])
    Z = np.array([0])

    dZ = A.relu.backward(dA, Z)

    assert(dZ.shape == Z.shape)
    assert_allclose(dZ, np.array([0]))
