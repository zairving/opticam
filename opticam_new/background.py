from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS
from typing import Callable, Union, Tuple
from astropy.stats import SigmaClip
from numpy.typing import ArrayLike

from opticam_new.helpers import get_data


class Background:
    
    def __init__(self, box_size: Union[int, Tuple[int, int]], sigma: float = 3, bkg_estimator: Callable = SExtractorBackground(), bkgrms_estimator: Callable = StdBackgroundRMS()):
        """
        Helper class for computing 2D backgrounds.

        Parameters
        ----------
        box_size : Union[int, Tuple[int, int]]
            Size of the background mesh.
        sigma : float
            How many standard deviations to use for sigma clipping.
        bkg_estimator : Callable
            Background estimator. It is recommended to use photutils background estimators (e.g.,
            `photutils.background.SExtractorBackground()`), but custom estimators can be used.
        bkgrms_estimator : Callable
            Background RMS estimator. It is recommended to use photutils background RMS estimators (e.g.,
            `photutils.background.StdBackgroundRMS()`), but custom estimators can be used.
        """
        
        self.box_size = box_size
        self.sigma_clip = SigmaClip(sigma=sigma)
        self.bkg_estimator = bkg_estimator
        self.bkgrms_estimator = bkgrms_estimator
    
    def __call__(self, image: Union[ArrayLike, str]) -> Background2D:
        """
        Compute the 2D background for file.
        
        Parameters
        ----------
        file : Union[ArrayLike, str]
            Image or directory path to an image.
        
        Returns
        -------
        Background2D
            2D image background.
        """
        
        if isinstance(image, str):
            data = get_data(image)
        else:
            data = image
        
        return Background2D(data, self.box_size, sigma_clip=self.sigma_clip, bkg_estimator=self.bkg_estimator, bkgrms_estimator=self.bkgrms_estimator)