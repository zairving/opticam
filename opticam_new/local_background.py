from photutils.aperture import ApertureStats, CircularAnnulus, EllipticalAnnulus
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple
from astropy.stats import SigmaClip


class LocalBackground(ABC):
    """
    Base class for local background estimators.
    """
    
    @abstractmethod
    def __call__(self, data: NDArray, error: NDArray, semimajor_axis: float, semiminor_axis: float,
                 theta: float, position: Tuple[float, float]) -> Tuple[float, float]:
        pass


class CircularLocalBackground(LocalBackground):
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
            The inner radius of the annulus (in units of the aperture radius).
        r_out : float
            The outer radius of the annulus (in units of the aperture radius).
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
    
    
    def __call__(self, data: NDArray, error: NDArray, semimajor_axis: float, semiminor_axis: float,
                 theta: float, position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute the local background and its error at a given position (*per pixel*).
        
        Parameters
        ----------
        data : NDArray
            The image data.
        error : NDArray
            The error in the image data.
        semimajor_axis : float
            The semi-major axis of the aperture.
        semiminor_axis : float
            The semi-minor axis of the aperture.
        psf_theta : float
            The rotation angle of the PSF.
        position : ArrayLike[float, float]
            The x, y position at which to compute the local background.
        
        Returns
        -------
        Tuple[float, float]
            The local background and its error per pixel.
        """
        
        psf_radius = max(semimajor_axis, semiminor_axis)
        
        annulus = CircularAnnulus(position, self.r_in_scale * psf_radius, self.r_out_scale * psf_radius)
        stats = ApertureStats(data, annulus, error=error, sigma_clip=self.sigma_clip)
        
        if self.method == 'mean':
            return stats.mean, stats.std
        else:
            return stats.median, stats.mad_std


class EllipticalLocalBackground(LocalBackground):
    """
    Compute the local background using an elliptical annulus.
    """
    
    def __init__(self, a_in_scale: float = 1, a_out_scale: float = 2, b_in_scale: float = 1, b_out_scale: float = 2,
                 method: Literal['mean', 'median'] = 'median', sigma_clip: SigmaClip = SigmaClip(maxiters=10)):
        """
        Local background estimator using an elliptical annulus.
        
        Parameters
        ----------
        a_in_scale : float, optional
            The inner semi-major axis of the aperture (in units of aperture semi-major axis), by default 1.
        a_out_scale : float, optional
            The outer semi-major axis of the aperture (in units of aperture semi-major axis), by default 2.
        b_in_scale : float, optional
            The inner semi-minor axis of the aperture (in units of aperture semi-minor axis), by default 1.
        b_out_scale : float, optional
            The outer semi-minor axis of the aperture (in units of aperture semi-minor axis), by default 2.
        method : Literal['mean', 'median'], optional
            The method to use to compute the local background. If 'mean', the mean and standard deviation of the pixel
            values in the annulus are used. If 'median', the median and the median absolute deviation are used,
            by default 'median'.
        sigma_clip : SigmaClip, optional
            The sigma clipper for removing outlier pixels in the annulus, by default SigmaClip(maxiters=10).
        """
        
        assert method in ['mean', 'median'], "[OPTICAM] method attribute of EllipticalLocalBackground() must be either 'mean' or 'median'."
        
        self.a_in_scale = a_in_scale
        self.a_out_scale = a_out_scale
        self.b_in_scale = b_in_scale
        self.b_out_scale = b_out_scale
        self.method = method
        self.sigma_clip = sigma_clip
    
    def __call__(self, data: NDArray, error: NDArray, semimajor_axis: float, semiminor_axis: float,
                 theta: float, position: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute the local background and its error at a given position (*per pixel*).
        
        Parameters
        ----------
        data : NDArray
            The image data.
        error : NDArray
            The error in the image data.
        semimajor_axis : float
            The semi-major axis of the aperture.
        semiminor_axis : float
            The semi-minor axis of the aperture.
        psf_theta : float
            The rotation angle of the PSF.
        position : ArrayLike[float, float]
            The x, y position at which to compute the local background.
        
        Returns
        -------
        Tuple[float, float]
            The local background and its error per pixel.
        """
        
        annulus = EllipticalAnnulus(position, self.a_in_scale * semimajor_axis, self.a_out_scale * semimajor_axis,
                                    self.b_out_scale * semiminor_axis, self.b_in_scale * semiminor_axis, theta)
        stats = ApertureStats(data, annulus, error=error, sigma_clip=self.sigma_clip)
        
        if self.method == 'mean':
            return stats.mean, stats.std
        else:
            return stats.median, stats.mad_std