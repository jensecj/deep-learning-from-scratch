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

def relu(Z):
    """Return the ReLU of Z."""
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)

    return A

def relu_deriv(dA, Z):
    """Return the derivative of the ReLU of Z, with respect to dA."""
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.  the
    # the derivative is 1 if dZ[i] > 0, otherwise it is 0. with a special case
    # for i = 0, which is set to 0 here.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ
