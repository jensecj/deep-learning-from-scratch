import os
import pickle
import random

import numpy as np

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

def split_dataset(ratio, dataset):
    assert len(dataset) >= 10, "The dataset is too small to split properly."
    assert len(dataset) > len(ratio), "Unable to split dataset with too many ratios."

    for r in ratio:
        assert r > 0, "Split ratio cannot be less than 0"

    assert np.isclose(sum(ratio), 1), "Split ratios should sum to 1."

    shuffled = random.shuffle(dataset)

    sizes = [ int(len(dataset) * r) for r in ratio ]
    sets = []

    for i in range(len(sizes)):
        s = dataset[:sizes[i]]
        dataset = dataset[sizes[i]:]

        sets.append(s)

    assert dataset == [], "Split has leftover data"

    return sets
