from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS
from typing import Callable, Union, Tuple, Dict
from astropy.stats import SigmaClip
from numpy.typing import ArrayLike, NDArray
from abc import ABC, abstractmethod


class Background(ABC):
    
    def __init__(self, box_size: Union[int, Tuple[int, int]], sigma_clip: SigmaClip = SigmaClip(sigma=3, maxiters=10),
                 bkg_estimator: Callable = SExtractorBackground(), bkgrms_estimator: Callable = StdBackgroundRMS()):
        """
        Helper class for computing 2D backgrounds.

        Parameters
        ----------
        box_size : Union[int, Tuple[int, int]]
            Size of the background mesh.
        sigma_clip : SigmaClip
            Sigma clipper.
        bkg_estimator : Callable
            Background estimator. It is recommended to use photutils background estimators (e.g.,
            `photutils.background.SExtractorBackground()`), but custom estimators can be used.
        bkgrms_estimator : Callable
            Background RMS estimator. It is recommended to use photutils background RMS estimators (e.g.,
            `photutils.background.StdBackgroundRMS()`), but custom estimators can be used.
        """
        
        self.box_size = box_size
        self.sigma_clip = sigma_clip
        self.bkg_estimator = bkg_estimator
        self.bkgrms_estimator = bkgrms_estimator
    
    def __call__(self, data: NDArray) -> Background2D:
        """
        Compute the 2D background for file.
        
        Parameters
        ----------
        data : NDArray
            Image data.
        
        Returns
        -------
        Background2D
            2D image background.
        """
        
        return Background2D(data, self.box_size, sigma_clip=self.sigma_clip, bkg_estimator=self.bkg_estimator,
                            bkgrms_estimator=self.bkgrms_estimator)