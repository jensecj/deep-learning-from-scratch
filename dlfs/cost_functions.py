import numpy as np

def squared_errors(predictions, labels):
    """
    SSE - Sum of Squared Errors.
    """
    return np.sum(np.square(predictions - labels))

def mean_squared_errors(predictions, labels):
    """
    MSE - Mean Squared Errors.
    """
    return np.mean(squared_errors(predictions, labels))

def root_mean_squared_errors(predictions, labels):
    """
    RMSE - Root Mean Squared Errors.
    """
    return np.sqrt(mean_squared_errors(predictions, labels))

def absolute_errors(predictions, labels):
    """
    SAE - Sum of Absolute Errors
    """
    return np.sum(np.abs(predictions - labels))

def mean_absolute_errors(predictions, labels):
    """
    MAE - Mean Absolute Errors.
    """
    return np.mean(absolute_errors(predictions, labels))

def cross_entropy(predictions, labels, epsilon=1e-12) -> Any:
    """
    Cross-Entropy, also known as Log-Loss.

    Input:
    predictions -- an ndarray of floats with values in [0,1].
    labels -- an ndarray of floats, the target values

    Returns:
    cost - a float, the cross entropy loss of the predictions compared to the target values.
    """
    assert(predictions.shape == labels.shape)

    # to avoid having to deal with the extreme cases of 0 and 1, we clip each
    # prediction to a small value larger than 0, and less than 1.
    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    assert np.all([predictions > 0]), "values in predictions cannot be less or equal than 0."
    assert np.all([predictions < 1]), "values in predictions cannot be greater than or equal 1."

    assert predictions.shape[0] == 1, "predictions was not a row vector."

    m = predictions.shape[1] # number of predictions
    # print(f"num predictions: {m}")

    log_prop = labels * np.log(predictions)
    # print(log_prop)
    inv_log_prop = (1 - labels) * np.log(1 - predictions)
    # print(inv_log_prop)
    log_sums = log_prop + inv_log_prop
    # print(log_sums)
    cost = np.sum(log_sums)
    # print(cost)
    cost = cost / m
    # print(cost)
    cost = -cost
    # print(cost)

    # cost = -(1.0/m) * np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
    # cost = -np.average(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    cost = np.clip(cost, epsilon, cost)

    assert(isinstance(cost, float))

    return cost
