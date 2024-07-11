from ctypes import Union
from photutils.segmentation import SourceFinder
from typing import Literal, Union, Dict
from numpy.typing import ArrayLike

from opticam_new.helpers import get_data


class CrowdedFinder:
    
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
    
    def __call__(self, image: Union[ArrayLike, str], threshold: float) -> SourceFinder:
        
        if isinstance(image, str):
            data = get_data(image)
        else:
            data = image
        
        segment_map = self.finder(data, threshold)
        segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        return segment_map

    def get_input_dict(self) ->  Dict:
        
        params_dict = {
            'npixels': self.npixels,
            'connectivity': self.connectivity,
            'nlevels': self.nlevels,
            'contrast': self.contrast,
            'mode': self.mode,
            'border_width': self.border_width
        }
        
        return params_dict


class Finder:
    
    def __init__(self, npixels: int = None, connectivity: Literal[4, 8] = 8, nlevels: int = 32, contrast: float = 0.001,
                 mode: Literal['exponential', 'linear', 'sinh'] = 'exponential', border_width: int = 0):
        
        self.npixels = npixels
        self.connectivity = connectivity
        self.border_width = border_width
        
        self.finder = SourceFinder(npixels=self.npixels, connectivity=self.connectivity, deblend=False,
                                   progress_bar=False)
    
    def __call__(self, image: Union[ArrayLike, str], threshold: float) -> SourceFinder:
        
        if isinstance(image, str):
            data = get_data(image)
        else:
            data = image
        
        segment_map = self.finder(data, threshold)
        segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        return segment_map

    def get_input_dict(self) ->  Dict:
        
        params_dict = {
            'npixels': self.npixels,
            'connectivity': self.connectivity,
            'border_width': self.border_width
        }
        
        return params_dict