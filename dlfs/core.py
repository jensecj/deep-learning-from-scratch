import os
import pickle

from dataclasses import dataclass
from typing import Any, List

@dataclass(frozen=True)
class Layer:
    hidden_units: int
    activation: Any = None

# TODO: fns for dimensions / num_layers?
@dataclass(frozen=True)
class Network:
    layers: List[Layer]
    cost: Any
    optimizer: Any

    def dimensions(self):
        """
        Return a list of the number of hidden units in each layer of the network.
        """
        return [ l.hidden_units for l in self.layers ]

def save_model(net, parameters, filename):
    """Save NET and PARAMETERS to FILENAME."""
    model = net, parameters
    with open(filename, 'wb') as f:
        return pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    """Load NET and PARAMETERS from FILENAME."""
    if not os.path.exists(filename):
        return None, None

    with open(filename, 'rb') as f:
        return pickle.load(f)
