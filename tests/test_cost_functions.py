import pytest
import numpy as np
from numpy.testing import assert_allclose

import dlfs.cost_functions as C

##################
# squared errors #
##################

def test_squared_errors_given_empty_predictions_and_empty_labels_should_raise_assertionerror():
    predictions = np.array([])
    labels = np.array([])

    with pytest.raises(AssertionError):
        cost = C.squared_errors(predictions, labels)

def test_squared_errors_given_column_vector_should_raise_assertionerror():
    predictions = np.array([[0.342],
                            [0.9372],
                            [0.01243]])
    labels = np.array([[0],
                       [1],
                       [0]])

    with pytest.raises(AssertionError):
        cost = C.squared_errors(predictions, labels)

def test_squared_errors_given_differently_shaped_predictions_and_labels_should_raise_assertionerror():
    predictions = np.array([[0.342, 0.9372, 0.01243]])
    labels = np.array([[0, 1]])

    with pytest.raises(AssertionError):
        cost = C.squared_errors(predictions, labels)

def test_squared_errors_given_zero_predictions_and_zero_labels_should_return_zero():
    predictions = np.array([0])
    labels = np.array([0])

    cost = C.squared_errors(predictions, labels)

    assert_allclose(cost, 0)

def test_squared_errors_given_zero_predictions_and_1_labels_should_return_1():
    predictions = np.array([0])
    labels = np.array([1])

    cost = C.squared_errors(predictions, labels)

    assert_allclose(cost, 1)

def test_squared_errors_given_negative_predictions_and_negative_labels():
    pass

def test_squared_errors_given_positive_predictions_and_negative_labels():
    pass

def test_squared_errors_given_negative_predictions_and_positive_labels():
    pass

def test_squared_errors_given_positive_predictions_and_negative_labels():
    pass

def test_squared_errors_given_row_vector():
    pass


#######################
# mean squared errors #
#######################

############################
# root mean squared errors #
############################

###################
# absolute errors #
###################

########################
# mean absolute errors #
########################

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

    # FIXME: is this correct?
    # assert_allclose(cost, 0.407714666667)
