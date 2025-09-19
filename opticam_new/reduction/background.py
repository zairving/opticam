from photutils.background import Background2D
from typing import Tuple
from numpy.typing import NDArray


class DefaultBackground:
    """
    Default background estimator.
    """
    
    def __init__(self, box_size: int | Tuple[int, int]):
        """
        Default background estimator.
        
        Parameters
        ----------
        box_size : int | Tuple[int, int]
            Size of the background mesh "pixels". If an integer is provided, the mesh pixels are squares of size
            `box_size` x `box_size`. If a tuple is provided, the mesh pixels are rectangles of width `box_size[1]` and
            height `box_size[0]`.
        """
        
        if not isinstance(box_size, int):
            assert len(box_size) == 2, "[OPTICAM] Incompatible box_size parameter passed to DefaultBackground. box_size must be either an integer or an interable of dimensions (e.g., [height, width])."
        
        self.box_size = box_size
    
    def __call__(self, data: NDArray) -> Background2D:
        """
        Compute the 2D background for an image.
        
        Parameters
        ----------
        data : NDArray
            Image data.
        
        Returns
        -------
        Background2D
            2D background.
        """
        
        return Background2D(data, self.box_size)