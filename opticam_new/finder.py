from ctypes import Union
from photutils.segmentation import SourceFinder
from typing import Literal, Union, Dict
from numpy.typing import ArrayLike, NDArray




class Finder:
    """
    Base class for source finders.
    """
    
    def __init__(self, npixels: int, connectivity: Literal[4, 8] = 8, nlevels: int = 32, contrast: float = 0.001,
                 mode: Literal['exponential', 'linear', 'sinh'] = 'exponential', border_width: int = 0):
        """
        Base class for source finders.
        
        Parameters
        ----------
        npixels : int
            The minimum number of connected source pixels.
        connectivity : Literal[4, 8], optional
            The source pixel connectivity, by default 8. If npixels=4 does not count diagonal pixels, npixels=8 does.
        nlevels : int, optional
            The number of threshold levels, by default 32.
        contrast : float, optional
            _description_, by default 0.001
        mode : Literal[&#39;exponential&#39;, &#39;linear&#39;, &#39;sinh&#39;], optional
            _description_, by default 'exponential'
        border_width : int, optional
            _description_, by default 0
        """
        
        self.npixels = npixels
        self.connectivity = connectivity
        self.border_width = border_width
        
        self.finder = SourceFinder(npixels=self.npixels, connectivity=self.connectivity, deblend=False,
                                   progress_bar=False)
    
    def __call__(self, data: NDArray, threshold: float) -> SourceFinder:
        
        segment_map = self.finder(data, threshold)
        
        if self.border_width > 0:
            segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        return segment_map


class CrowdedFinder(Finder):
    
    def __init__(self, npixels: int = None, connectivity: Literal[4, 8] = 8, nlevels: int = 32, contrast: float = 0.001,
                 mode: Literal['exponential', 'linear', 'sinh'] = 'exponential', border_width: int = 0):
        
        self.npixels = npixels
        self.connectivity = connectivity
        self.nlevels = nlevels
        self.contrast = contrast
        self.mode = mode
        self.border_width = border_width
        
        self.finder = SourceFinder(npixels=self.npixels, connectivity=self.connectivity, nlevels=self.nlevels,
                                   contrast=self.contrast, mode=self.mode, progress_bar=False)


