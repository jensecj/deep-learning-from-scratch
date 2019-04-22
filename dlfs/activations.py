import numpy as np

class linear:
    @staticmethod
    def forward(Z):
        return Z, Z

    @staticmethod
    def backward(dA, Z):
        dZ = dA * 1

        assert (dZ.shape == Z.shape)

        return dZ

class sigmoid:
    @staticmethod
    def forward(Z):
        """
        Vectorized sigmoid of numpy array Z.
        sigmoid(z) = 1 / (1 + e^-z)
        """
        A = 1 / (1 + np.exp(-Z))

        assert(A.shape == Z.shape)

        return A, Z

    @staticmethod
    def backward(dA, Z):
        """Return the derivative of sigmoid of Z, with respect to dA."""
        s, _ = sigmoid.forward(Z);
        dZ = dA * s * (1-s)

        assert (dZ.shape == Z.shape)

        return dZ

class tanh:
    @staticmethod
    def forward(Z):
        """
        Return the hyperbolic tangent of Z.
        """
        # A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        A = np.tanh(Z)

        assert(A.shape == Z.shape)

        return A, Z

    @staticmethod
    def backward(dA, Z):
        A, _ = tanh.forward(Z)
        dZ = dA * (1 - np.power(A, 2))

        assert (dZ.shape == Z.shape)

        return dZ

class relu:
    @staticmethod
    def forward(Z):
        """Return the ReLU of Z."""
        A = np.maximum(0, Z)

        assert(A.shape == Z.shape)

        return A, Z

    @staticmethod
    def backward(dA, Z):
        """Return the derivative of the ReLU of Z, with respect to dA."""
        d = np.array(dA, copy=True) # just converting dz to a correct object.  the
        # the derivative is 1 if dZ[i] > 0, otherwise it is 0. with a special case
        # for i = 0, which is set to 0 here.
        d[Z <= 0] = 0
        d[Z > 0] = 1

        dZ = dA * d

        assert (dZ.shape == Z.shape)

        return dZ

class leaky_relu:
    @staticmethod
    def forward(Z, epsilon = 0.01):
        """Return the Leaky ReLU of Z."""
        A = np.maximum(epsilon*Z, Z)

        assert(A.shape == Z.shape)

        return A, Z

    @staticmethod
    def backward(dA, Z, epsilon = 0.01):
        """Return the derivative of the Leaky ReLU of Z, with respect to dA."""
        d = np.array(dA, copy=True) # just converting dz to a correct object.  the
        # the derivative is 1 if dZ[i] > 0, otherwise it is 0. with a special case
        # for i = 0, which is set to 0 here.
        d[Z <= 0] = epsilon
        d[Z > 0] = 1

        dZ = dA * d

        assert (dZ.shape == Z.shape)

        return dZ

class softmax:
    def forward(Z):
        A = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

        assert(A.shape == Z.shape)

        return A, Z

    def backward(dA, Z):
        s = Z.reshape(-1,1)
        sm = np.diagflat(s) - np.dot(s, s.T)

        dZ = dA * sm

        assert (dZ.shape == Z.shape)

        return dZ
