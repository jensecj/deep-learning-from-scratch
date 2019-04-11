import numpy as np
import matplotlib.pyplot as plt

import dlfs.plotting as plot
import dlfs.network as nn
from dlfs.network import Layer, Network
import dlfs.activations as A
import dlfs.cost_functions as C
import dlfs.optimizers as O

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m / 2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m, D)) # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j, N * (j + 1))
        t = np.linspace(j*3.12, (j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

inputs, labels = load_planar_dataset()

print(inputs.shape)
print(labels.shape)

input_layers = inputs.shape[0]

layers = [
    Layer(input_layers, A.tanh),
    Layer(5, A.tanh),
    Layer(1, A.sigmoid)
]

net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

parameters, costs = nn.train(net, inputs, labels, 1.2, 10000)
predictions = nn.predict(net, inputs, parameters)
predictions = np.round(predictions)

accuracy = np.sum((predictions == labels) / inputs.shape[1])

print(f"accuracy: {accuracy*100:.3}%")

plot.decision_boundary(lambda x: np.round(nn.predict(net, x.T, parameters)), inputs, labels)
plt.title(f"Decision Boundary for network with {len(layers)} layers.")
plt.show()
