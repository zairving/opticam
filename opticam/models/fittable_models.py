import numpy as np
from numpy.typing import NDArray


def gaussian(
    x: NDArray,
    amplitude: float,
    mu: float,
    std: float,
    ) -> NDArray:
    """
    One dimensional Gaussian function.
    
    Parameters
    ----------
    x : NDArray
        The function inputs.
    amplitude : float
        The height of the function peak.
    mu : float
        The mean of the Gaussian.
    std : float
        The standard deviation of the Gaussian.
    
    Returns
    -------
    NDArray
        The Gaussian function evaluated at the inputs.
    """
    
    return amplitude * np.exp(-.5 * ((x - mu) / std)**2)