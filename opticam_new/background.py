from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS
from typing import Callable, Union, Tuple, Dict
from astropy.stats import SigmaClip
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod

from opticam_new.helpers import get_data


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
    
    def get_input_dict(self) ->  Dict:
        
        params_dict = {
            "box_size": self.box_size,
        }
        
        for key, value in self.sigma_clip.__dict__.items():
            if not key.startswith("_"):
                params_dict["SigmaClip " + str(key)] = value
        
        params_dict.update({
            "bkg_estimator": self.bkg_estimator.__class__.__name__,
            "bkgrms_estimator": self.bkgrms_estimator.__class__.__name__,
            })

        return params_dict