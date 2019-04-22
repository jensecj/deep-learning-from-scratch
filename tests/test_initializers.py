import numpy as np
from numpy.testing import assert_allclose

import dlfs.initializers as I

def test_zeros_initializer():
    a = I.zeros((1,1))
    assert_allclose(a, np.array([[0]]))

    b = I.zeros((1,2))
    assert_allclose(b, np.array([[0, 0]]))

    c = I.zeros((2,2))
    assert_allclose(c, np.array([[0,0],
                                 [0,0]]))

    d = I.zeros((3,4))
    assert_allclose(d, np.array([[0,0,0,0],
                                 [0,0,0,0],
                                 [0,0,0,0]]))

def test_ones_initializer():
    a = I.ones((1,1))
    assert_allclose(a, np.array([[1]]))

    b = I.ones((1,2))
    assert_allclose(b, np.array([[1, 1]]))

    c = I.ones((2,2))
    assert_allclose(c, np.array([[1,1],
                                 [1,1]]))

    d = I.ones((3,4))
    assert_allclose(d, np.array([[1,1,1,1],
                                 [1,1,1,1],
                                 [1,1,1,1]]))

def test_constant_initializer():
    a = I.constant((1,1), 0.01)
    print(a)
    assert_allclose(a, np.array([[0.01]]))

    b = I.constant((1,2), 4)
    print(b)
    assert_allclose(b, np.array([[4, 4]]))

    c = I.constant((2,2), 0.1234)
    print(c)
    assert_allclose(c, np.array([[0.1234,0.1234],
                                 [0.1234,0.1234]]))

    d = I.constant((3,4), 9)
    print(d)
    assert_allclose(d, np.array([[9,9,9,9],
                                 [9,9,9,9],
                                 [9,9,9,9]]))

def test_random():
    pass

def test_variance_scaled():
    pass

def test_xavier():
    pass

def test_xavier_normal():
    pass

def test_he():
    pass
