from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class Layer:
    hidden_units: int
    activation: str

@dataclass
class Network:
    layers: List[Layer]
    cost: str
    optimizer: str

def initialize(net: Network):
    weights = {}
    biases = {}

    network_dimensions = [ d.hidden_units for d in [ l for l in net.layers]]
    num_layers = len(network_dimensions)

    print("network has {0} layers".format(num_layers))
    print("network dimensions: {0}".format(network_dimensions))

    for l in range(0, num_layers):
        weights[l] = np.random.randn(network_dimensions[l], network_dimensions[l-1]) / np.sqrt(network_dimensions[l-1]) * 0.01
        biases[l] = np.zeros((network_dimensions[l], 1))

    return weights, biases
