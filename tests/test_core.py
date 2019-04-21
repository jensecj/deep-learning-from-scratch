import pytest
import numpy as np
from numpy.testing import assert_allclose

from dlfs.core import split_dataset

def test_split_dataset_with_small_dataset_should_error():
    dataset = [1,2,3]
    ratio = (0.7, 0.2, 0.1)

    with pytest.raises(AssertionError):
        train, dev, test = split_dataset(ratio, dataset)

def test_split_dataset_with_too_low_ratio_should_error():
    dataset = [1,2,3,4,5,6,7,8,9,10]
    ratio = (0.5, 0.2, 0.1)

    with pytest.raises(AssertionError):
        train, dev, test = split_dataset(ratio, dataset)

def test_split_dataset_with_too_high_ratio_should_error():
    dataset = [1,2,3,4,5,6,7,8,9,10]
    ratio = (0.5, 1.2, 3.1)

    with pytest.raises(AssertionError):
        train, dev, test = split_dataset(ratio, dataset)

def test_split_dataset_with_even_ratio():
    dataset = [1,2,3,4,5,6,7,8,9,10]
    ratio = (0.5, 0.5)
    a,b = split_dataset(ratio, dataset)

    assert len(a) == len(b) == 5

    for n in a:
        assert not n in b, "Split contains duplicates"

    print(a)
    print (b)

def test_split_dataset_splits_properly():
    dataset = [1,2,3,4,5,6,7,8,9,10]
    ratio = (0.7, 0.2, 0.1)
    train, dev, test = split_dataset(ratio, dataset)

    assert len(train) == 7
    assert len(dev) == 2
    assert len(test) == 1

    print(train)
    print(dev)
    print(test)
