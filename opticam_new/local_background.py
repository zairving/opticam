from photutils.aperture import ApertureStats, CircularAnnulus, EllipticalAnnulus
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple
from astropy.stats import SigmaClip



class CircularLocalBackground:
    """
    Compute the local background using a circular annulus.
    """
    
    def __init__(self, r_in_scale: float = 1, r_out_scale: float = 2, method: Literal['mean', 'median'] = 'median',
                 sigma_clip: SigmaClip = SigmaClip(maxiters=10)):
        """
        Local background estimator using a circular annulus.
        
        Parameters
        ----------
        r_in : float
            The inner radius of the annulus (in units of the PSF radius).
        r_out : float
            The outer radius of the annulus (in units of the PSF radius).
        method : Literal['mean', 'median']
            The method to use to compute the local background. If 'mean', the mean and standard deviation of the pixel
            values in the annulus are used. If 'median', the median and the median absolute deviation are used.
        sigma_clip : SigmaClip, optional
            The sigma clipper for removing outlier pixels in the annulus, by default SigmaClip(maxiters=10)
        """
        
        assert method in ['mean', 'median'], "[OPTICAM] method attribute of CircularLocalBackground() must be either 'mean' or 'median'."
        
        self.r_in_scale = r_in_scale
        self.r_out_scale = r_out_scale
        self.method = method
        self.sigma_clip = sigma_clip
    
    def __call__(self, data: NDArray, error: NDArray, psf_radius: float,
                 position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute the local background and its error at a given position (*per pixel*).
        
        Parameters
        ----------
        data : NDArray
            The image data.
        psf_radius : float
            The PSF radius.
        position : ArrayLike[float, float]
            The x, y position at which to compute the local background.
        
        Returns
        -------
        Tuple[float, float]
            The local background and its error per pixel.
        """
        
        annulus = CircularAnnulus(position, self.r_in_scale * psf_radius, self.r_out_scale * psf_radius)
        stats = ApertureStats(data, annulus, error=error, sigma_clip=self.sigma_clip)
        
        if self.method == 'mean':
            return stats.mean, stats.std
        else:
            return stats.median, stats.mad_std
