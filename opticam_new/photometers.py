from abc import ABC, abstractmethod
from logging import Logger
from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from photutils.aperture import aperture_photometry, EllipticalAperture
from photutils.segmentation import SourceCatalog, detect_threshold

from opticam_new.background.global_background import BaseBackground
from opticam_new.background.local_background import BaseLocalBackground
from opticam_new.correctors.flat_field_corrector import FlatFieldCorrector
from opticam_new.finders import DefaultFinder
from opticam_new.utils.constants import fwhm_scale
from opticam_new.utils.fits_handlers import get_data


class BasePhotometer(ABC):
    """
    Base class for performing photometry on OPTICAM catalogues.
    """

    def __init__(
        self,
        match_sources: bool = True,
        source_matching_tolerance: float = 2.,
        local_background_estimator: None | BaseLocalBackground = None,
        ):
        """
        Initialise a photometer.
        
        Parameters
        ----------
        match_sources : bool, optional
            Whether to match sources in the image to sources in the catalogue, by default True. This can improve source
            position matching, but may lead to incorrect source matching if the field is crowded or the
            alignments are poor. If False, the photometer will use the source positions from the catalogue directly,
            and the directory in which the resulting light curves will be saved will have a 'forced' prefix.
        source_matching_tolerance : float, optional
            The tolerance for source position matching in standard deviations (assuming a Gaussian PSF), by default 2.
            This parameter defines how far from the transformed catalogue position a source can be while still being
            considered the same source.
        local_background_estimator : None | BaseLocalBackground, optional
            The local background estimator to use, by default `None`. If `None`, the catalogue's 2D background estimator
            is used. If not `None`, this will be used instead of the catalogue's 2D background estimator.
        """
        
        self.match_sources = match_sources
        self.source_matching_tolerance = source_matching_tolerance
        self.scale_factor = 1 if match_sources else 2  # use a larger aperture if source matching is disabled
        
        if local_background_estimator is not None:
            assert callable(local_background_estimator), "[OPTICAM] local_background_estimator must be either None or a callable object."
        
        self.local_background_estimator = local_background_estimator

    @abstractmethod
    def compute(
        self,
        image: NDArray,
        image_err: NDArray,
        source_coords: NDArray,
        image_coords: None | NDArray,
        psf_params: Dict[str, float],
        ) -> Dict[str, List]:
        """
        Perform photometry on the given image using the provided source coordinates and PSF parameters. If defining a
        custom photometer, this method must be implemented. The resulting dictionary must contain 'flux' and
        'flux_err' keys, as well as any additional metrics that the photometer computes. The time stamps for each
        image are handled by the catalogue, and so they do not need to be included in the results dictionary.
        
        Parameters
        ----------
        image : NDArray
            The image. If `local_background_estimator` is undefined, this image will be background subtracted.
        image_err : NDArray
            The error in the image.
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : None | NDArray
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in 
            *degrees*).
        
        Returns
        -------
        Dict[str, List]
            A dictionary containing the results of the photometry. The dictionary must contain 'flux' and 'flux_error'
            keys, as well as any additional metrics that the photometer computes. The time stamps for each image are
            handled by the catalogue, and so they do not need to be included in the results dictionary.
        """
        
        pass


class SimplePhotometer(BasePhotometer):
    """
    A simple photometer that provides simple aperture photometry routines with support for local background estimations
    using annuli.
    """

    def compute(
        self,
        image: NDArray,
        image_err: NDArray,
        source_coords: NDArray,
        image_coords: None | NDArray,
        psf_params: Dict[str, float],
        ) -> Dict[str, List]:
        """
        Compute the simple photometry for the given image using the provided source coordinates and PSF parameters.
        
        Parameters
        ----------
        image : NDArray
            The image. If `local_background_estimator` is undefined, this image will be background subtracted.
        image_err : NDArray
            The error in the image.
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : None | NDArray
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Dict[str, List]
            The results of the photometry
        """
        
        results = self.define_results_dict()
        
        for i in range(len(source_coords)):
            
            # get position of source depending on whether source matching is enabled or not
            position = self.get_position(
                source_coords,
                image_coords,
                i,
                psf_params,
                )
            
            # if position is None, pad the results dictionary and continue to the next source
            if position is None:
                results = self.pad_results_dict(results)
                continue
            
            # populate the results dictionary with the computed flux, flux error, and background (if applicable)
            results = self.populate_results_dict(
                results,
                self.compute_aperture_flux,
                image,
                image_err,
                position,
                psf_params,
                )
        
        return results

    def compute_aperture_flux(
        self,
        data: NDArray,
        error: NDArray,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> Tuple[float, float] | Tuple[float, float, float, float]:
        """
        Compute the aperture flux of a source in the image.
        
        Parameters
        ----------
        data : NDArray
            The image.
        error : NDArray
            The error in the image.
        position : NDArray
            The position of the source.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Tuple[float, float] | Tuple[float, float, float, float]
            The flux and flux error. If `local_background_estimator` is defined, the background and its error are also
            returned.
        """
        
        aperture = EllipticalAperture(
            position,
            fwhm_scale * self.scale_factor * psf_params['semimajor_sigma'],
            fwhm_scale * self.scale_factor * psf_params['semiminor_sigma'],
            psf_params['orientation'],
            )
        
        phot_table = aperture_photometry(data, aperture, error=error)
        
        if self.local_background_estimator is None:
            return phot_table["aperture_sum"].value[0], phot_table["aperture_sum_err"].value[0]
        else:
            aperture_area = aperture.area_overlap(data)  # aperture area in pixels
            
            # estimate local background in the annulus
            local_background_per_pixel, local_background_error_per_pixel = self.local_background_estimator(
                data,
                error,
                position,
                psf_params['semimajor_sigma'],
                psf_params['semiminor_sigma'],
                psf_params['orientation'],
                )
            
            # estimate the total background in aperture
            total_bkg = local_background_per_pixel * aperture_area
            total_bkg_error = np.sqrt(local_background_error_per_pixel**2 * aperture_area)
            
            flux = float(phot_table["aperture_sum"].value[0] - total_bkg)
            flux_error = float(np.sqrt(phot_table["aperture_sum_err"].value[0]**2 + total_bkg_error**2))
            local_background = float(total_bkg)
            local_background_errors = float(total_bkg_error)
            
            return flux, flux_error, local_background, local_background_errors

    def get_position(
        self,
        source_coords: NDArray,
        image_coords: NDArray | None,
        source_index: int,
        psf_params: Dict[str, float],
        ) -> NDArray | None:
        """
        Get the position of a source in an image.
        
        Parameters
        ----------
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image.
        source_index : int
            The source index.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        NDArray
            The source coordinates.
        """
        
        if self.match_sources:
            return self.get_closest_source(
                source_coords,
                image_coords,
                source_index,
                psf_params,
                )
        else:
            return source_coords[source_index]

    def get_closest_source(
        self,
        source_coords: NDArray,
        image_coords: NDArray | None,
        source_index: int,
        psf_params: Dict[str, float],
        ) -> NDArray | None:
        """
        Given a source, find the closest source in the catalogue.
        
        Parameters
        ----------
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image.
        source_index : int
            The source index.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        NDArray | None
            The coordinates of the closest source.
        """
        
        if image_coords is None:
            return None
        
        # get distances between sources and initial position
        distances = np.sqrt((image_coords[:, 0] - source_coords[source_index][0])**2 + (image_coords[:, 1] - source_coords[source_index][1])**2)
        
        # if the closest source is further than the specified tolerance
        if np.min(distances) > self.source_matching_tolerance * np.sqrt(psf_params['semimajor_sigma']**2 + psf_params['semiminor_sigma']**2):
            return None
        else:
            # get the position of the closest source (assumed to be the source of interest)
            return image_coords[np.argmin(distances)]

    def define_results_dict(
        self,
        ) -> Dict[str, List]:
        """
        Define a results dictionary for the photometer depending on whether `local_background_estimator` is defined.
        
        Returns
        -------
        Dict[str, List]
            The results dictionary with keys 'flux', 'flux_error'. If `local_background_estimator` is defined, the
            dictionary will also contain 'background' and 'background_error'.
        """
        
        results = {
            'flux': [],
            'flux_err': [],
        }
        
        if self.local_background_estimator is not None:
            results['background'] = []
            results['background_err'] = []
        
        return results

    def pad_results_dict(
        self,
        results: Dict[str, List],
        ) -> Dict[str, List]:
        """
        Pad the results dictionary with None values for flux and flux error, and background and background error if
        `local_background_estimator' is defined. This is used when a source cannot be matched or its position is
        invalid.
        
        Parameters
        ----------
        results : Dict[str, List]
            The results dictionary to pad.
        
        Returns
        -------
        Dict[str, List]
            The padded results dictionary.
        """
        
        results['flux'].append(None)
        results['flux_err'].append(None)
        
        if self.local_background_estimator is not None:
            results['background'].append(None)
            results['background_err'].append(None)
        
        return results

    def populate_results_dict(
        self,
        results: Dict[str, List],
        phot_function: Callable,
        image: NDArray,
        image_err: NDArray,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> Dict[str, List]:
        """
        Populate the results dictionary with the computed flux, flux error, and background (if applicable) using the
        provided photometry function.
        
        Parameters
        ----------
        results : Dict[str, List]
            The results dictionary to populate.
        phot_function : Callable
            The photometry function to use for computing the flux and flux error. This function should take the image,
            image error, position, and PSF parameters as arguments and return the flux and flux error, and optionally
            the background and background error if `local_background_estimator` is defined.
        image : NDArray
            The image.
        image_err : NDArray
            The error in the image.
        position : NDArray
            The position of the source in the image.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Dict[str, List]
            The updated results dictionary with the computed flux, flux error, and background (if applicable).
        """
        
        if self.local_background_estimator is None:
            flux, flux_err = phot_function(
                image,
                image_err,
                position,
                psf_params,
            )
            
            results['flux'].append(flux)
            results['flux_err'].append(flux_err)
        else:
            flux, flux_err, background, background_err = phot_function(
                image,
                image_err,
                position,
                psf_params,
            )
            
            results['flux'].append(flux)
            results['flux_err'].append(flux_err)
            results['background'].append(background)
            results['background_err'].append(background_err)
        
        return results


class OptimalPhotometer(SimplePhotometer):
    """
    A photometer that implements the optimal photometry method described in Naylor 1998, MNRAS, 296, 339-346.
    """

    def compute(
        self,
        image: NDArray,
        image_err: NDArray,
        source_coords: NDArray,
        image_coords: None | NDArray,
        psf_params: Dict[str, float],
        ) -> Dict[str, List]:
        """
        Compute the optimal photometry for each source in the image using the method described in Naylor 1998, MNRAS,
        296, 339-346.
        
        Parameters
        ----------
        image : NDArray
            The image. If `local_background_estimator` is undefined, this image will be background subtracted.
        image_err : NDArray
            The error in the image.
        source_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : None | NDArray
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Dict[str, List]
            The results of the photometry, including 'flux', 'flux_error', and optionally 'background' and
            'background_error' if `local_background_estimator` is defined.
        """
        
        results = self.define_results_dict()
        
        for i in range(len(source_coords)):
            
            position = self.get_position(
                source_coords,
                image_coords,
                i,
                psf_params,
            )
            
            if position is None:
                results = self.pad_results_dict(results)
                continue
            
            results = self.populate_results_dict(
                results,
                self.compute_optimal_flux,
                image,
                image_err,
                position,
                psf_params,
                )
        
        return results

    def compute_optimal_flux(
        self,
        image: NDArray,
        error: NDArray,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> Tuple[float, float] | Tuple[float, float, float, float]:
        """
        Compute the optimal flux of a source in the image as described in Naylor 1998, MNRAS, 296, 339-346.
        
        Parameters
        ----------
        image : NDArray
            The image.
        error : NDArray
            The error in the image.
        position : NDArray
            The position of the source in the image, given as (y, x) coordinates.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_sigma' (in pixels), 'semiminor_sigma' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        Tuple[float, float] | Tuple[float, float, float, float]
            The flux and flux error. If `local_background_estimator` is defined, the background and its error are also
            returned.
        """
        
        # define pixel coordinates
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        
        # convert orientation from degrees to radians
        theta = psf_params['orientation'] * np.pi / 180
        
        # offset coordinates to the position of the source and align axes with the orientation of the PSF
        x0, y0 = position
        x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
        y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
        
        # compute the weights for each pixel using a 2D Gaussian
        weights = np.exp(- .5 * ((x_rot / psf_params['semimajor_sigma'])**2 + (y_rot / psf_params['semiminor_sigma'])**2))
        weights /= np.sum(weights)  # normalise weights
        
        # compute optimal flux and its error
        flux = np.sum(image * weights)
        flux_error = np.sqrt(np.sum((error * weights)**2))
        
        if self.local_background_estimator is None:
            return flux, flux_error
        else:
            # estimate local background using annulus
            local_background_per_pixel, local_background_error_per_pixel = self.local_background_estimator(
                image,
                error,
                position,
                psf_params['semimajor_sigma'],
                psf_params['semiminor_sigma'],
                psf_params['orientation'],
                )
            
            # estimate the total background in aperture
            total_bkg = local_background_per_pixel * np.sum(weights)
            total_bkg_error = np.sqrt(local_background_error_per_pixel**2 * np.sum(weights))
            
            flux = flux - total_bkg
            flux_error = np.sqrt(flux_error**2 + total_bkg_error**2)
            local_background = total_bkg
            local_background_errors = total_bkg_error
            
            return flux, flux_error, local_background, local_background_errors




def perform_photometry(
    file: str,
    photometer: BasePhotometer,
    source_coords: NDArray,
    gains: Dict[str, float],
    bmjds: Dict[str, float],
    flat_corrector: FlatFieldCorrector | None,
    rebin_factor: int,
    remove_cosmic_rays: bool,
    background: BaseBackground,
    threshold: float,
    finder: DefaultFinder,
    psf_params: Dict[str, Dict[str, float]],
    fltr: str,
    logger: Logger,
    ) -> Dict[str, List]:
    
    image, error = get_data(
        file=file,
        gain=gains[file],
        flat_corrector=flat_corrector,
        rebin_factor=rebin_factor,
        return_error=True,
        remove_cosmic_rays=remove_cosmic_rays,
        )
    
    if photometer.local_background_estimator is None:
        bkg = background(image)  # get 2D background
        image = image - bkg.background  # remove background from image
        error = np.sqrt(error**2 + bkg.background_rms**2)  # propagate error
        threshold = threshold * bkg.background_rms  # define source detection threshold
    else:
        # estimate source detection threshold from noisy image
        threshold = detect_threshold(image, threshold, error=error)  # type: ignore
    
    image_coords = None  # assume no image coordinates by default
    if photometer.match_sources:
        try:
            segm = finder(image, threshold)
            tbl = SourceCatalog(image, segm).to_table()
            image_coords = np.array([tbl["xcentroid"].value,
                                    tbl["ycentroid"].value]).T
        except Exception as e:
            logger.warning(f"[OPTICAM] Could not determine source coordinates in {file}: {e}")
    
    results = photometer.compute(image, error, source_coords, image_coords, psf_params[fltr])
    
    assert 'flux' in results, f"[OPTICAM] Photometer {photometer.__class__.__name__}'s compute method must return a 'flux' key."
    assert 'flux_err' in results, f"[OPTICAM] Photometer {photometer.__class__.__name__}'s compute method must return a 'flux_err' key."
    
    # results check
    for key, values in results.items():
        for i, value in enumerate(values):
            if value is None:
                logger.warning(f"[OPTICAM] {key} could not be determined for source {i + 1} in {fltr} (got value {value}).")
    
    # add time stamp
    results['BMJD'] = bmjds[file]  # add time of observation
    
    return results





