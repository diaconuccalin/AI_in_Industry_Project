import numpy as np


def softplus(x, beta=1, threshold=20):
    # Adapted from the PyTorch implementation of Softplus
    return np.where(x * beta <= threshold, (1 / beta) * np.log(1 + np.exp(beta * x)), x * beta)


def soft_truncation(x, noise):
    # Adapted from Microsoft's causica library
    # https://github.com/microsoft/causica/blob/b3c79a01f30f44ed36c582ffe2b4522058d82a73/causica/data_generation/csuite/simulate.py
    return softplus(x + 1) + softplus(0.5 + noise) - 3.0


def reverse_soft_truncation(x, noise):
    # Adapted from Microsoft's causica library
    # https://github.com/microsoft/causica/blob/b3c79a01f30f44ed36c582ffe2b4522058d82a73/causica/data_generation/csuite/simulate.py
    return softplus(-x + 1.5) + softplus(0.5 + noise) - 3.0
