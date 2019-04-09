import numpy as np
from numpy.testing import assert_allclose

from dlfs.cost_functions import *

def test_cross_entropy():
    labels = np.asarray([[1, 1, 1]])
    predictions = np.array([[.8, .9, 0.4]])

    assert_allclose(cross_entropy(predictions, labels), [0.41493159961539694])
