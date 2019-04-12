import pytest
import numpy as np
from numpy.testing import assert_allclose

import dlfs.cost_functions as C

#################
# cross entropy #
#################

def test_cross_entropy_with_zero_predictions_zero_labels():
    predictions = np.array([[0, 0]])
    labels = np.array([[0, 0]])

    cost = C.cross_entropy(predictions, labels)
    assert_allclose(cost, 1e-12)

def test_cross_entropy_with_column_vector():
    predictions = np.array([[.8],
                            [.9],
                            [0.4]])
    labels = np.array([[1],
                       [1],
                       [1]])

    with pytest.raises(AssertionError):
        cost = C.cross_entropy(predictions, labels)

def test_cross_entropy_with_row_vector():
    predictions = np.array([[.8, .9, 0.4]])
    labels = np.array([[1, 1, 1]])

    cost = C.cross_entropy(predictions, labels)

    assert_allclose(cost, 0.41493159961539694)

def test_cross_entropy_with_predictions_greater_than_one():
    predictions = np.array([[.8, 1.2, 1.4]])
    labels = np.array([[1, 1, 1]])

    cost = C.cross_entropy(predictions, labels)

    assert_allclose(cost, 0.07438118377)

def test_cross_entropy_with_predictions_less_than_0():
    predictions = np.array([[.8, -0.532, 1.4]])
    labels = np.array([[1, 1, 1]])

    cost = C.cross_entropy(predictions, labels)

    assert_allclose(cost, 0.407714666667)
