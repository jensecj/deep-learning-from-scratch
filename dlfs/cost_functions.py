import numpy as np

def cross_entropy(predictions, labels):
    """
    Also known as log-loss.
    """
    assert(predictions.shape == labels.shape)
    m = labels.shape[1]

    cost = (1./m) * (-np.dot(labels, np.log(predictions).T) - np.dot(1-labels, np.log(1-predictions).T))
    cost = np.squeeze(cost)

    assert(cost.shape == ())

    return cost
