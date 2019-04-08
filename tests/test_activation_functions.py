import numpy as np
from numpy.testing import assert_allclose

import dlfs.activation_functions as A

def test_sigmoid_with_negative():
    activation, activation_cache = A.sigmoid(np.array([-1]))
    assert_allclose(activation, np.array([0.26894142]), rtol=1e-8, atol=0)

def test_sigmoid_with_positive():
    activation, activation_cache = A.sigmoid(np.array([1]))
    assert_allclose(activation, np.array([0.7310585786]), rtol=1e-8, atol=0)

def test_sigmoid_with_zero():
    activation, activation_cache = A.sigmoid(np.array([0]))
    assert_allclose(activation, np.array([0.5]), rtol=1e-8, atol=0)

def test_sigmoid_row_and_column_vectors():
    activation, activation_cache = A.sigmoid(np.array([1, 2, 3]))
    assert_allclose(activation, np.array([0.73105858, 0.88079708, 0.95257413]), rtol=1e-8, atol=0)

    activation, activation_cache = A.sigmoid(np.array([[1.283462, 0.023864, 2.00981624]]))
    assert_allclose(activation, np.array([[0.78303851, 0.50596572, 0.8818238738]]), rtol=1e-8, atol=0)

def test_relu_with_negative():
    activation, activation_cache = A.relu(np.array([-1]))
    assert_allclose(activation, np.array([0]), rtol=1e-8, atol=0)

def test_relu_with_positive():
    activation, activation_cache = A.relu(np.array([1]))
    assert_allclose(activation, np.array([1]), rtol=1e-8, atol=0)

def test_relu_with_zero():
    activation, activation_cache = A.relu(np.array([0]))
    assert_allclose(activation, np.array([0]), rtol=1e-8, atol=0)

def test_relu_row_and_column_vectors():
    activation, activation_cache = A.relu(np.array([-5, 2, -2, 11]))
    assert_allclose(activation, np.array([0, 2, 0, 11]), rtol=1e-8, atol=0)

    activation, activation_cache = A.relu(np.array([[1.283462, 0.023864, 2.00981624]]))
    assert_allclose(activation, np.array([[1.283462, 0.023864, 2.00981624]]), rtol=1e-8, atol=0)
