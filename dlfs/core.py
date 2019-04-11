import os
import pickle

from dataclasses import dataclass
from typing import Any, List

@dataclass(frozen=True)
class Layer:
    hidden_units: int
    activation: Any

# TODO: fns for dimensions / num_layers?
@dataclass
class Network:
    layers: List[Layer]
    cost: Any
    optimizer: Any

def save_model(net, parameters, filename):
    model = net, parameters
    with open(filename, 'wb') as f:
        return pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    if not os.path.exists(filename):
        return None, None

    with open(filename, 'rb') as f:
        return pickle.load(f)
