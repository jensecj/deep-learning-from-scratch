import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model

import dlfs.plotting as plot
import dlfs.network as nn
from dlfs.core import Layer, Network
import dlfs.cost_functions as C
import dlfs.optimizers as O
import dlfs.activations as A

inputs, labels = sklearn.datasets.make_blobs(n_samples=200, random_state=5, n_features=2, centers=6)

# shape input so its fits out model
inputs = inputs.T
labels = labels.reshape(1, labels.shape[0])
labels = labels % 2 # blobs are binary

layers = [Layer(inputs.shape[0]),
          Layer(4, A.tanh),
          Layer(1, A.sigmoid)]
net = Network(layers, C.cross_entropy, O.batch_gradient_descent)

iterations = 1500
learning_rate = 0.5

parameters, costs = nn.train(net, inputs, labels, learning_rate, iterations)

predictions = nn.predict(net, inputs, parameters)

accuracy = np.sum((predictions == labels) / inputs.shape[1])
print(f"Accuracy: {(accuracy*100):.3}%")

plot.decision_boundary(lambda x: nn.predict(net, x.T, parameters), inputs, labels)
plt.title(f"Decision Boundary for network with {len(net.layers)} layers.\nAccuracy: {(accuracy*100):.3}%")
plt.show()

plot.costs(costs)
plt.title(f"Costs for {iterations} iterations with learning rate = {learning_rate}")
plt.show()
