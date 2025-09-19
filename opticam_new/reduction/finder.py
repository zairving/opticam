from photutils.segmentation import SourceFinder, SegmentationImage
from numpy.typing import NDArray


class DefaultFinder:
    """
    Default source finder. Combines image segmentation with source deblending.
    """
    
    def __init__(self, npixels: int, border_width: int = 0):
        """
        Default source finder. Combines image segmentation with source deblending.
        
        Parameters
        ----------
        npixels : int
            The minimum number of connected source pixels.
        border_width : int, optional
            Sources within this many pixels of the border will be ignored, by default 0 (no sources are ignored).
        """
        
        assert type(npixels) is int and npixels > 0, '[OPTICAM] npixels must be a positive integer.'
        
        self.border_width = border_width
        self.finder = SourceFinder(npixels=npixels, progress_bar=False)
    
    def __call__(self, data: NDArray, threshold: float) -> SegmentationImage:
        
        segment_map = self.finder(data, threshold)
        
        if self.border_width > 0:
            segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        return segment_map


