import numpy as np

def sigmoid(Z):
    """
    Vectorized sigmoid of numpy array Z.
    sigmoid(z) = 1 / (1 + e^-z)
    """
    A = 1 / (1 + np.exp(-Z))

    assert(A.shape == Z.shape)

    return A

def sigmoid_deriv(dA, Z):
    """Return the derivative of sigmoid of Z, with respect to dA."""
    s = sigmoid(Z);
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ
