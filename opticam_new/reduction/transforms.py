import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.transform import SimilarityTransform

def find_translation(
    coords: NDArray,
    reference_coords: NDArray
    ) -> SimilarityTransform:
    """
    Find the translation that maps `coords` onto `reference_coords`.
    
    Parameters
    ----------
    coords : NDArray
        The source coordinates.
    reference_coords : NDArray
        The reference source coordinates.
    
    Returns
    -------
    SimilarityTransform
        The transformation matrix that maps `coords` onto `reference_coords`.
    """
    
    distance_matrix = cdist(reference_coords, coords)
    reference_indices, indices = linear_sum_assignment(distance_matrix)
    
    dx = np.mean(reference_coords[reference_indices, 0] - coords[indices, 0])
    dy = np.mean(reference_coords[reference_indices, 1] - coords[indices, 1])
    
    return SimilarityTransform(translation=[dx, dy])