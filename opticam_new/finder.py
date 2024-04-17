from ctypes import Union
from photutils.segmentation import SourceFinder
from typing import Literal, Union
from numpy.typing import ArrayLike

from opticam_new.helpers import get_data


class Finder:
    
    def __init__(self, npixels: int = None, connectivity: Literal[4, 8] = 8, nlevels: int = 32, contrast: float = 0.001, border_width: int = 0):
        
        self.border_width = border_width
        
        self.finder = SourceFinder(npixels=npixels, connectivity=connectivity, nlevels=nlevels, contrast=contrast, progress_bar=False)
    
    def __call__(self, image: Union[ArrayLike, str], threshold: float) -> SourceFinder:
        
        if isinstance(image, str):
            data = get_data(image)
        else:
            data = image
        
        segment_map = self.finder(data, threshold)
        segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        return segment_map


