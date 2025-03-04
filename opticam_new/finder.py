from photutils.segmentation import SourceFinder, SegmentationImage
from numpy.typing import NDArray


class Finder:
    """
    Default source finder.
    """
    
    def __init__(self, npixels: int = None, border_width: int = None):
        """
        Default source finder.
        
        Parameters
        ----------
        npixels : int, optional
            The minimum number of connected source pixels, by default 128 / W**2, where W is the image width.
        border_width : int, optional
            The minimum distance from the image border for sources to be indentified, by default 1/16th of the image
            width.
        """
        
        self.border_width = border_width
        
        if npixels is not None:
            self.finder = SourceFinder(npixels, deblend=False, progress_bar=False)
        else:
            self.finder = None
    
    def __call__(self, data: NDArray, threshold: float) -> SegmentationImage:
        
        if self.finder is None:
            self.finder = SourceFinder(int(128 / (2048 / data.shape[0])**2), deblend=False, progress_bar=False)
        
        if self.border_width is None:
            self.border_width = data.shape[0] // 16
        
        segment_map = self.finder(data, threshold)
        
        if self.border_width > 0:
            segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        return segment_map


class CrowdedFinder:
    """
    Crowded source finder. Similar to `Finder`, but with source deblending.
    """
    
    def __init__(self, npixels: int = None, border_width: int = None):
        """
        Crowded source finder. Similar to `Finder`, but with source deblending.
        
        Parameters
        ----------
        npixels : int, optional
            The minimum number of connected source pixels, by default 128 / W**2, where W is the image width.
        border_width : int, optional
            The minimum distance from the image border for sources to be indentified, by default 1/16th of the image
            width.
        """
        
        self.border_width = border_width
        
        if npixels is not None:
            self.finder = SourceFinder(npixels=npixels, deblend=True, progress_bar=False)
        else:
            self.finder = None
    
    def __call__(self, data: NDArray, threshold: float) -> SegmentationImage:
        
        if self.finder is None:
            self.finder = SourceFinder(int(128 / (2048 / data.shape[0])**2), deblend=True, progress_bar=False)
        
        if self.border_width is None:
            self.border_width = data.shape[0] // 16
        
        segment_map = self.finder(data, threshold)
        
        if self.border_width > 0:
            segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        return segment_map


