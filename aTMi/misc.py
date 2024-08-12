import numpy as np

def binomial_combinations_k2(n:int)->np.ndarray:
    assert n > 1, "Parameter n > 1, otherwise no binomial combinations."
    a = np.array(np.arange(0, n))
    i, j = np.triu_indices(len(a), 1)
    combinations = np.stack([a[i], a[j]]).T
    return combinations