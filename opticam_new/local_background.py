from photutils.aperture import ApertureStats, CircularAnnulus, EllipticalAnnulus
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Tuple
from astropy.stats import SigmaClip


class LocalBackground(ABC):
    """
    Base class for local background estimators.
    """
    
    def __init__(self, r_in_scale: float = 1, r_out_scale: float = 2, sigma_clip: SigmaClip = SigmaClip(maxiters=10)):
        """
        Local background estimator using an elliptical annulus.
        
        Parameters
        ----------
        r_in_scale : float, optional
            The inner axes of the annulus (in units of aperture semi-major/semi-minor axes or radius), by default 1.
        r_out_scale : float, optional
            The outer axes of the annulus (in units of aperture semi-major/semi-minor axes or radius), by default 2.
        sigma_clip : SigmaClip, optional
            The sigma clipper for removing outlier pixels in the annulus, by default SigmaClip(maxiters=10).
        """
        
        self.r_in_scale = r_in_scale
        self.r_out_scale = r_out_scale
        self.sigma_clip = sigma_clip
    
    @abstractmethod
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
        theta : float
            The rotation angle of the PSF.
        position : Tuple[float, float]
            The x, y position at which to compute the local background.
        
        Returns
        -------
        Tuple[float, float]
            The local background and its error per pixel.
        """
        
        pass


class CircularLocalBackground(LocalBackground):
    """
    Compute the local background using a circular annulus.
    """
    
    
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
        theta : float
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
        
        return stats.mean, stats.std


class EllipticalLocalBackground(LocalBackground):
    """
    Compute the local background using an elliptical annulus.
    """
    
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
        theta : float
            The rotation angle of the PSF.
        position : ArrayLike[float, float]
            The x, y position at which to compute the local background.
        
        Returns
        -------
        Tuple[float, float]
            The local background and its error per pixel.
        """
        
        annulus = EllipticalAnnulus(position, self.r_in_scale * semimajor_axis, self.r_out_scale * semimajor_axis,
                                    self.r_out_scale * semiminor_axis, self.r_in_scale * semiminor_axis, theta)
        stats = ApertureStats(data, annulus, error=error, sigma_clip=self.sigma_clip)
        
        return stats.mean, stats.std