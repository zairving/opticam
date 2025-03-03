from photutils.background import Background2D
from typing import Union, Tuple
from numpy.typing import NDArray


class Background:
    """
    Default background estimator.
    """
    
    def __init__(self, box_size: Union[int, Tuple[int, int]] = None):
        """
        Default background estimator.
        
        Parameters
        ----------
        box_size : Union[int, Tuple[int, int]], optional
            Size of the background mesh. If None, the box size is set to 1/16th of the image width.
        """
        
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
        
        # set box_size to 1/16th of the image size if not specified
        if self.box_size is None:
            self.box_size = data.shape[0] // 16
        
        return Background2D(data, self.box_size)