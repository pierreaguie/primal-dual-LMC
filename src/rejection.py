import numpy as np

def rejection_sampling(n_samples : int, m : np.ndarray, C : np.ndarray):
    """ 
    Samples n_samples points from the 2D trucated standard Gaussian distrbution of mean m, supported by the ellipse of centered at 0 and of covaraince matrix C, using rejection sampling.
    """

    samples = np.zeros((n_samples, 2))
    idx = 0

    while idx < n_samples:
        x = m + np.random.randn(2)
        if x.T @ C @ x <= 1:
            samples[idx] = np.copy(x)
            idx += 1

    return samples