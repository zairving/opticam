import os

try:
    os.environ['OMP_NUM_THREADS'] = '1'  # set number of threads to 1 for better multiprocessing performance
except:
    pass

from tqdm.contrib.concurrent import process_map  # process_map removes a lot of the boilerplate from multiprocessing
from tqdm import tqdm
from astropy.table import QTable
import numpy as np
from photutils.segmentation import SourceCatalog
from photutils.aperture import aperture_photometry, CircularAperture, EllipticalAperture
from photutils.utils import calc_total_error
from skimage.transform import matrix_transform
from matplotlib import pyplot as plt
from functools import partial
from typing import Literal, Tuple
from numpy.typing import ArrayLike, NDArray
import csv
from abc import ABC, abstractmethod

from opticam_new.catalog import Catalog


class BasePhotometer(ABC):
    """
    Base class for photometry methods in OPTICAM.
    """

    def __init__(self, cat: Catalog):
        """
        Initialise the photometer with a catalog.
        
        Parameters
        ----------
        cat : Catalog
            The source catalog used for photometry.
        """
        
        self.cat = cat

    @abstractmethod
    def __call__(self, overwrite: bool = False, *args, **kwargs) -> None:
        """
        Perform photometry on the source catalog. This method should be implemented by subclasses.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing light curves, by default False.
        *args : tuple
            Additional positional arguments for specific photometry methods.
        """
        
        pass


class Photometer(BasePhotometer):
    """
    Perform normal and optimal photometry on a source catalog.
    """
    
    def __call__(self, overwrite: bool = False, background_method: Literal['global', 'local'] = 'global',
                tolerance: float = 5.) -> None:
        """
        Perform photometry by fitting for the source positions in each image. In general, this method should produce
        light curves with better signal-to-noise ratios than forced photometry. However, this method can misidentify
        sources if the field is crowded or the alignments are poor.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing light curves, by default False.
        background_method : Literal['global', 'local'], optional
            The method to use for background subtraction, by default 'global'. 'global' uses the background attribute to
            compute the 2D background across an entire image, while 'local' uses the local_background attribute to
            estimate the local background around each source.
        tolerance : float, optional
            The tolerance for source position matching in standard deviations (assuming a Gaussian PSF), by default 5.
            This parameter defines how far from the transformed catalog position a source can be while still being
            considered the same source. If the alignments are good and/or the field is crowded, consider reducing this
            value. For poor alignments and/or uncrowded fields, this value can be increased.
        """
        
        # create output directories if they do not exist
        if not os.path.isdir(self.cat.out_directory + f"normal_light_curves"):
            os.mkdir(self.cat.out_directory + f"normal_light_curves")
        if not os.path.isdir(self.cat.out_directory + f"optimal_light_curves"):
            os.mkdir(self.cat.out_directory + f"optimal_light_curves")
        
        for fltr in list(self.cat.catalogs.keys()):
            # skip cameras with no images
            if len(self.cat.camera_files[fltr]) == 0:
                continue
            
            # get list of possible light curve files
            light_curve_files = [self.cat.out_directory + f"normal_light_curves/{fltr}_source_{i + 1}.csv" for i in range(len(self.cat.catalogs[fltr]))]
            light_curve_files += [self.cat.out_directory + f"optimal_light_curves/{fltr}_source_{i + 1}.csv" for i in range(len(self.cat.catalogs[fltr]))]
            
            # check if light curves already exist
            if all([os.path.isfile(file) for file in light_curve_files]) and not overwrite:
                self.cat.logger.info(f'[OPTICAM] {fltr} light curves already exist and overwrite is False. Skipping ...')
                continue
            
            # get PSF parameters
            semimajor_sigma = self.cat.aperture_selector(self.cat.catalogs[fltr]["semimajor_sigma"].value)
            semiminor_sigma = self.cat.aperture_selector(self.cat.catalogs[fltr]["semiminor_sigma"].value)
            
            self.cat.logger.info(f'[OPTICAM] {fltr} semi-major axis of PSF: {semimajor_sigma} pixels ({semimajor_sigma * self.cat.binning_scale * self.cat.rebin_factor * self.cat.pix_scales[fltr]}").')
            self.cat.logger.info(f'[OPTICAM] {fltr} semi-minor axis of PSF: {semiminor_sigma} pixels ({semiminor_sigma * self.cat.binning_scale * self.cat.rebin_factor * self.cat.pix_scales[fltr]}").')
            self.cat.logger.info(f'[OPTICAM] {fltr} normal aperture semi-major axis: {self.cat.fwhm_scale * semimajor_sigma} pixels ({self.cat.fwhm_scale * semimajor_sigma * self.cat.binning_scale * self.cat.rebin_factor * self.cat.pix_scales[fltr]}").')
            self.cat.logger.info(f'[OPTICAM] {fltr} normal aperture semi-minor axis: {self.cat.fwhm_scale * semiminor_sigma} pixels ({self.cat.fwhm_scale * semiminor_sigma * self.cat.binning_scale * self.cat.rebin_factor * self.cat.pix_scales[fltr]}").')
            
            chunksize = max(1, len(self.cat.camera_files[fltr]) // 100)  # chunk size for parallel processing (must be >= 1)
            results = process_map(partial(self._perform_photometry, fltr=fltr, semimajor_sigma=semimajor_sigma,
                                          semiminor_sigma=semiminor_sigma, background_method=background_method,
                                          tolerance=tolerance), self.cat.camera_files[fltr],
                                  max_workers=self.cat.number_of_processors,
                                  desc=f"[OPTICAM] Performing photometry on {fltr} images",
                                  disable=not self.cat.verbose, chunksize=chunksize)
            
            self._save_results(results, fltr)

    def _perform_photometry(self, file: str, fltr: str, semimajor_sigma: float, semiminor_sigma: float,
                             background_method: Literal['global', 'local'], tolerance: float):
        """
        Perform photometry on a single of image.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        fltr : str
            The filter.
        semimajor_sigma : float
            The semimajor axis of the PSF.
        semiminor_sigma : float
            The semiminor axis of the PSF.
        background_method : Literal['global', 'local']
            The method to use for background subtraction.
        tolerance : float
            The tolerance for source position matching in standard deviations.
        phot_type : Literal['normal', 'optimal', 'both']
            The type of photometry to perform.
        
        Returns
        -------
        Tuple
            The photometric results.
        """
        
        # if file does not have a transform, and it's not the reference image, skip it
        if file not in self.cat.transforms.keys() and file != self.cat.camera_files[fltr][self.cat.reference_indices[fltr]]:
            return None, None, None, None, None, None, np.zeros(len(self.cat.catalogs[fltr]))
        
        # define lists to store results for each file
        normal_fluxes, normal_flux_errors = [], []
        optimal_fluxes, optimal_flux_errors = [], []
        detections = np.zeros(len(self.cat.catalogs[fltr]))
        
        # get image data
        data, error = self.cat.get_data(file, return_error=True)
        
        # get background subtracted image for source detection
        bkg = self.cat.background(data)
        clean_data = data - bkg.background
        
        if background_method == 'global':
            # combine error in background subtracted image
            error = np.sqrt(error**2 + bkg.background_rms**2)
        
        # find sources in the background subtracted image
        try:
            segment_map = self.cat.finder(clean_data, threshold=self.cat.threshold * bkg.background_rms)
        except:
            # if no sources are found, return None for all results
            return None, None, None, None, None, None, np.zeros(len(self.cat.catalogs[fltr]))
        
        # create source table
        file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
        file_tbl = file_cat.to_table()
        
        # for each source in the catalog
        for i in range(len(self.cat.catalogs[fltr])):
            # locate source in the source table
            try:
                position = self._get_position_of_nearest_source(file_tbl, i, fltr, file, tolerance)
            except:
                # if source is not found, append None to results
                normal_fluxes.append(None)
                normal_flux_errors.append(None)
                optimal_fluxes.append(None)
                optimal_flux_errors.append(None)
                continue
            
            # if source is found, increment the detection counter
            detections[i] += 1
            
            # perform photometry
            if background_method == 'global':
                # compute normal flux using global background
                flux, flux_error = self._compute_normal_flux(clean_data, error, position, semimajor_sigma,
                                                              semiminor_sigma,
                                                              self.cat.catalogs[fltr]['orientation'][i].value)
            else:
                # compute normal flux using local background
                flux, flux_error = self._compute_normal_flux(data, error, position, semimajor_sigma, semiminor_sigma,
                                                             self.cat.catalogs[fltr]['orientation'][i].value, True)
            
            # append results to lists
            normal_fluxes.append(flux)
            normal_flux_errors.append(flux_error)
            
            if background_method == 'global':
                # compute optimal flux using global background
                flux, flux_error = self._compute_optimal_flux(clean_data, error, position, semimajor_sigma,
                                                               semiminor_sigma,
                                                               self.cat.catalogs[fltr]['orientation'][i].value)
            else:
                # compute optimal flux using local background
                flux, flux_error = self._compute_optimal_flux(data, error, position, semimajor_sigma, semiminor_sigma,
                                                               self.cat.catalogs[fltr]['orientation'][i].value, True)
            
            # append results to lists
            optimal_fluxes.append(flux)
            optimal_flux_errors.append(flux_error)
        
        return self.cat.bdts[file], normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, detections

    def _parse_results(self, results):
        """
        Parse the photometry results.
        
        Parameters
        ----------
        results :
            The photometry results.
        
        Returns
        -------
        Tuple
            The parsed photometric results.
        """
        
        bdts, normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, detections = zip(*results)
        return list(bdts), list(normal_fluxes), list(normal_flux_errors), list(optimal_fluxes), list(optimal_flux_errors), np.sum(detections, axis=0)

    def _save_results(self, results: Tuple, fltr: str) -> None:
        """
        Save the photometry results.
        
        Parameters
        ----------
        results : Tuple
            The photometry results.
        phot_type : str
            The type of photometry that has been performed.
        fltr : str
            The filter.
        """
        
        # parse results
        bdts, normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, detections = self._parse_results(results)
        
        # save light curves
        for source_index in tqdm(range(len(self.cat.catalogs[fltr])), disable=not self.cat.verbose, desc=f"[OPTICAM] Saving {fltr} light curves"):
            
            # normal
            with open(self.cat.out_directory + f"normal_light_curves/{fltr}_source_{source_index + 1}.csv", "w") as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(['TDB', "flux", "flux_error"])
                for i in range(len(bdts)):
                    try:
                        csvwriter.writerow([bdts[i], normal_fluxes[i][source_index],
                                            normal_flux_errors[i][source_index]])
                    except TypeError:
                        continue
            
            # optimal
            with open(self.cat.out_directory + f"optimal_light_curves/{fltr}_source_{source_index + 1}.csv", "w") as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(['TDB', "flux", "flux_error"])
                
                # for each observation in which a source was detected
                for i in range(len(bdts)):
                    try:
                        csvwriter.writerow([bdts[i], optimal_fluxes[i][source_index],
                                            optimal_flux_errors[i][source_index]])
                    except TypeError:
                        continue
        
        # plot number of detections per source
        self._plot_detections_per_source(detections, fltr)

    def _compute_normal_flux(self, data: NDArray, error: NDArray, position: NDArray, semimajor_sigma: float,
                              semiminor_sigma: float, orientation: float,
                              estimate_local_background: bool = False) -> Tuple[float, float]:
        """
        Compute the flux at a given position using simple aperture photometry.
        
        Parameters
        ----------
        clean_data : NDArray
            The image.
        error : NDArray
            The total error in the image.
        position : NDArray
            The aperture position.
        semimajor_sigma : float
            The semimajor axis of the (presumed 2D Gaussian) PSF.
        semiminor_sigma : float
            The semiminor axis of the (presumed 2D Gaussian) PSF.
        orientation : float
            The orientation of the (presumed 2D Gaussian) PSF.
        estimate_local_background : bool, optional
            Whether to estimate the local background. If True, the local background will be estimated using the
            local_background attribute, otherwise the data are assumed to be background subtracted.
        
        Returns
        -------
        Tuple[float, float]
            The flux and its error.
        """
        
        aperture = EllipticalAperture(position, self.cat.fwhm_scale * semimajor_sigma,
                                      self.cat.fwhm_scale * semiminor_sigma, orientation)
        phot_table = aperture_photometry(data, aperture, error=error)
        
        if estimate_local_background:
            local_background_per_pixel, local_background_error_per_pixel = self.cat.local_background(data, error, self.cat.scale * semimajor_sigma, self.cat.scale * semiminor_sigma, orientation, position)
            aperture_area = aperture.area_overlap(data)
            aperture_background, aperture_background_error = aperture_area * local_background_per_pixel, np.sqrt(aperture_area * local_background_error_per_pixel)
            return phot_table['aperture_sum'].value[0] - aperture_background, np.sqrt(phot_table['aperture_sum_err'].value[0]**2 + aperture_background_error**2)
        else:
            return phot_table["aperture_sum"].value[0], phot_table["aperture_sum_err"].value[0]

    def _compute_optimal_flux(self, data: NDArray, error: NDArray, position: NDArray, semimajor_sigma: float,
                               semiminor_sigma: float, orientation: float,
                               estimate_local_background: bool = False) -> Tuple[float, float]:
        """
        Compute the flux at a given position using the optimal photometry method of Naylor 1998, MNRAS, 296, 339.
        
        Parameters
        ----------
        data : NDArray
            The image.
        error : NDArray
            The total error in the image.
        position : NDArray
            The aperture position.
        semimajor_sigma : float
            The semimajor axis of the (presumed 2D Gaussian) PSF.
        semiminor_sigma : float
            The semiminor axis of the (presumed 2D Gaussian) PSF.
        orientation : float
            The orientation of the (presumed 2D Gaussian) PSF.
        estimate_local_background : bool
            Whether to estimate the local background. If True, the local background will be estimated using the
            local_background attribute, otherwise the data are assumed to be background subtracted.
        Returns
        -------
        Tuple[float, float]
            The flux and its error.
        """
        
        if estimate_local_background:
            # compute local background
            local_background_per_pixel, local_background_error_per_pixel = self.cat.local_background(data, error, self.cat.scale * semimajor_sigma, self.cat.scale * semiminor_sigma, orientation, position)
            clean_data = data - local_background_per_pixel  # subtract local background
            error = np.sqrt(error**2 + local_background_error_per_pixel**2)  # add local background error in quadrature
        else:
            # assume data are background subtracted
            clean_data = data
        
        # optimal photometry
        y, x = np.ogrid[:clean_data.shape[0], :clean_data.shape[1]]  # define pixel coordinates
        x0, y0 = position  # define source position
        x_rot = (x - x0) * np.cos(orientation) + (y - y0) * np.sin(orientation)  # align pixel coordinates with source orientation and shift source to origin
        y_rot = -(x - x0) * np.sin(orientation) + (y - y0) * np.cos(orientation)  # align pixel coordinates with source orientation and shift source to origin
        weights = np.exp(-0.5 * ((x_rot / semimajor_sigma)**2 + (y_rot / semiminor_sigma)**2))  # compute pixel weights assuming a 2D Gaussian PSF
        weights /= np.sum(weights)  # normalise weights
        
        return np.sum(clean_data * weights), np.sqrt(np.sum((error * weights)**2))

    def _get_position_of_nearest_source(self, file_tbl: QTable, source_index: int, fltr: str, file: str,
                                         tolerance: float) -> NDArray:
        """
        Get the position of the source nearest an expected source position in an image.
        
        Parameters
        ----------
        file_tbl : QTable
            The source catalog of the image.
        source_index : int
            The target source index.
        fltr : str
            The filter of the image.
        file : str
            The name of the image file.
        tolerance : float
            The tolerance for source position matching in standard deviations.
        
        Returns
        -------
        NDArray
            The position of the nearest source ([x, y]).
        
        Raises
        ------
        ValueError
            If no source is found close enough to the expected source position.
        """
        
        # get source position from catalog
        catalog_position = (self.cat.catalogs[fltr]["xcentroid"][source_index], self.cat.catalogs[fltr]["ycentroid"][source_index])
        
        # if file is the reference image
        if file == self.cat.camera_files[fltr][self.cat.reference_indices[fltr]]:
            # use the catalog position as the initial position
            initial_position = catalog_position
        else:
            # use the transformed catalog position as the initial position
            initial_position = matrix_transform(catalog_position, self.cat.transforms[file])[0]
        
        # get positions of sources
        positions = np.array([[file_tbl["xcentroid"][i], file_tbl["ycentroid"][i]] for i in range(len(file_tbl))])
        
        # get distances between sources and initial position
        distances = np.sqrt((positions[:, 0] - initial_position[0])**2 + (positions[:, 1] - initial_position[1])**2)
        
        # if the closest source is further than the specified tolerance
        if np.min(distances) > tolerance*np.sqrt(self.cat.catalogs[fltr]["semimajor_sigma"][source_index].value**2 + self.cat.catalogs[fltr]["semiminor_sigma"][source_index].value**2):
            raise ValueError(f"[OPTICAM] No source found close enough to source {source_index + 1} in {file}. This is usually due to poor alignments.")
        else:
            # get the position of the closest source (assumed to be the source of interest)
            return positions[np.argmin(distances)]

    def _plot_detections_per_source(self, detections: NDArray, fltr: str) -> None:
        """
        Plot the number of detections per source.
        
        Parameters
        ----------
        detections : NDArray
            The number of detections per source.
        fltr : str
            The filter used to observe the sources.
        """
        
        # save number of observations per source to file
        with open(self.cat.out_directory + f"diag/{fltr}_observations.csv", "w") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(["source", "observations"])
            
            # for each source
            for i in range(len(self.cat.catalogs[fltr])):
                csvwriter.writerow([i + 1, detections[i]])
        
        fig, ax = plt.subplots(tight_layout=True)
        
        ax.bar(np.arange(len(self.cat.catalogs[fltr])) + 1, detections, color="none", edgecolor="black")
        ax.axhline(len(self.cat.camera_files[fltr]), color="red", linestyle="--", lw=1)
        
        ax.set_xlabel("Source")
        ax.set_ylabel("Number of detections")
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        fig.savefig(self.cat.out_directory + f"diag/{fltr}_observations.png")
        
        if self.cat.show_plots:
            plt.show(fig)
        else:
            plt.close(fig)


class ForcedPhotometer(BasePhotometer):
    """
    Perform forced photometry on a source catalog.
    """

    def __call__(self, overwrite: bool = False) -> None:
        """
        Perform forced photometry on the source catalog. The light curves produced by this method are generally
        going to have lower signal-to-noise ratios than those produced by the photometry() method, but they have the
        benefit of being able to extract light curves for sources that are not detected in all images.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing light curves, by default False.
        """
        
        # create output directories if they do not exist
        if not os.path.isdir(self.cat.out_directory + f"aperture_light_curves"):
            os.mkdir(self.cat.out_directory + f"aperture_light_curves")
        if not os.path.isdir(self.cat.out_directory + f"annulus_light_curves"):
            os.mkdir(self.cat.out_directory + f"annulus_light_curves")
        
        # for each camera
        for fltr in list(self.cat.catalogs.keys()):
            # skip cameras with no images
            if len(self.cat.camera_files[fltr]) == 0:
                continue
            
            # get list of possible light curve files
            light_curve_files = [self.cat.out_directory + f"aperture_light_curves/{fltr}_source_{i}.csv" for i in range(len(self.cat.catalogs[fltr]))]
            light_curve_files += [self.cat.out_directory + f"annulus_light_curves/{fltr}_source_{i}.csv" for i in range(len(self.cat.catalogs[fltr]))]
            
            # check if light curves already exist
            if all([os.path.isfile(file) for file in light_curve_files]) and not overwrite:
                self.cat.logger.info(f'[OPTICAM] {fltr} light curves already exist and overwrite is False. Skipping ...')
                continue
            
            # get aperture radius
            try:
                radius = self.cat.scale*self.cat.aperture_selector(self.cat.catalogs[fltr]["semimajor_sigma"].value)
            except:
                # skip cameras with no sources
                continue
            
            self.cat.logger.info(f'[OPTICAM] {fltr} forced photometry aperture radius: {radius} pixels. ({radius * self.cat.binning_scale * self.cat.rebin_factor * self.cat.pix_scales[fltr]}").')
            
            chunksize = max(1, len(self.cat.camera_files[fltr]) // 100)  # chunk size for parallel processing (must be >= 1)
            results = process_map(partial(self._perform_forced_photometry, fltr=fltr, radius=radius),
                                  self.cat.camera_files[fltr],
                                  max_workers=self.cat.number_of_processors,
                                  desc=f"[OPTICAM] Performing forced photometry on {fltr} images",
                                  disable=not self.cat.verbose, chunksize=chunksize)
            
            self._save_results(results, fltr)

    def _perform_forced_photometry(self, file: str, fltr: str, radius: float):
        """
        Perform forced photometry on a batch of images.
        
        Parameters
        ----------
        batch : List[str]
            The batch of image names.
        fltr : str
            The camera filter.
        radius : float
            The aperture radius
        
        Returns
        -------
        Tuple
            The photometric results.
        """
        
        # define lists to store results for each file
        aperture_fluxes, aperture_flux_errors = [], []
        annulus_fluxes, annulus_flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
        
        # get image transform and determine quality flag
        if file == self.cat.camera_files[fltr][self.cat.reference_indices[fltr]]:
            flag = 'A'
        elif file not in self.cat.transforms.keys():
            flag = 'B'
        else:
            flag = 'A'
            transform = self.cat.transforms[file]
        
        # get image data and its error
        data, error = self.cat.get_data(file, return_error=True)
        
        # get background subtracted image and its error if required
        bkg = self.cat.background(data)
        clean_data = data - bkg.background
        clean_error = calc_total_error(clean_data, bkg.background_rms, error)
        
        # for each source
        for i in range(len(self.cat.catalogs[fltr])):
            # load the source's catalog position
            catalog_position = (self.cat.catalogs[fltr]["xcentroid"][i], self.cat.catalogs[fltr]["ycentroid"][i])
            
            # try to transform the catalog position using the image transform, otherwise use the catalog position
            try:
                position = matrix_transform(catalog_position, transform)[0]
            except:
                position = catalog_position
            
            # aperture
            flux, flux_error = self._compute_aperture_flux(clean_data, clean_error, position, radius)
            aperture_fluxes.append(flux)
            aperture_flux_errors.append(flux_error)
            # annulus
            flux, flux_error, local_background, local_background_error, local_background_per_pixel, local_background_error_per_pixel = self._compute_annulus_flux(data, error, position, radius)
            annulus_fluxes.append(flux)
            annulus_flux_errors.append(flux_error)
            local_backgrounds.append(local_background)
            local_background_errors.append(local_background_error)
            local_backgrounds_per_pixel.append(local_background_per_pixel)
            local_background_errors_per_pixel.append(local_background_error_per_pixel)
        
        return aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flag

    @staticmethod
    def _compute_aperture_flux(clean_data: NDArray, error: NDArray, position: ArrayLike,
                               radius: float) -> Tuple[float, float]:
        """
        Compute the flux and error for a given aperture position and radius.
        
        Parameters
        ----------
        clean_data : NDArray
            The background subtracted image.
        error : NDArray
            The error in the image.
        position : ArrayLike
            The aperture position.
        radius : float
            The aperture radius.
        
        Returns
        -------
        Tuple[float, float]
            The flux and error.
        """
        
        aperture = CircularAperture(position, r=radius)
        phot_table = aperture_photometry(clean_data, aperture, error=error)
        
        return phot_table["aperture_sum"].value[0], phot_table["aperture_sum_err"].value[0]

    def _compute_annulus_flux(self, data: NDArray, error: NDArray, position: ArrayLike,
                              radius: float) -> Tuple[float, float, float, float, float, float]:
        """
        Compute the local-background-subtracted flux and error for a given aperture position and radius.
        
        Parameters
        ----------
        data : ArrayLike
            The image data.
        error : ArrayLike
            The error in the image.
        position : ArrayLike
            The aperture position.
        radius : float
            The aperture radius.
        
        Returns
        -------
        Tuple[float, float, float, float, float, float]
            The flux, flux error, local background, local background error, local background per pixel, and local
            background error per pixel.
        """
        
        # define aperture
        aperture = CircularAperture(position, r=radius)
        aperture_area = aperture.area_overlap(data)  # aperture area in pixels
        phot_table = aperture_photometry(data, aperture, error=error)
        
        # estimate local background per pixel using circular annulus
        local_background_per_pixel, local_background_error_per_pixel = self.cat.local_background(data, error, radius, radius, 0, position)
        
        # calculate total background in aperture
        total_bkg = local_background_per_pixel * aperture_area
        total_bkg_error = np.sqrt(local_background_error_per_pixel * aperture_area)
        
        flux = phot_table["aperture_sum"].value[0] - total_bkg
        flux_error = np.sqrt(phot_table["aperture_sum_err"].value[0]**2 + total_bkg_error**2)
        local_background = total_bkg
        local_background_errors = total_bkg_error
        local_backgrounds_per_pixel = local_background_per_pixel
        local_background_errors_per_pixel = local_background_error_per_pixel
        
        return flux, flux_error, local_background, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel

    def _save_results(self, results: Tuple, fltr: str) -> None:
        """
        Unpack and save the forced photometry results.
        
        Parameters
        ----------
        results : Tuple
            The photometric results.
        phot_type : str
            The type of photometry that has been performed.
        fltr : str
            The filter.
        """
        
        # get image time stamps
        bdts = [self.cat.bdts[file] for file in self.cat.camera_files[fltr]]
        
        # save light curves
        # parse results
        aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = self._parse_results(results)
        
        # save light curves
        for source_index in tqdm(range(len(self.cat.catalogs[fltr])), disable=not self.cat.verbose, desc=f"[OPTICAM] Saving {fltr} light curves"):
            
            # aperture
            with open(self.cat.out_directory + f'aperture_light_curves/{fltr}_source_{source_index + 1}.csv', 'w') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(['TDB', 'flux', 'flux_error', 'quality_flag'])
                for i in range(len(self.cat.camera_files[fltr])):
                    csvwriter.writerow([bdts[i], aperture_fluxes[i][source_index],
                                        aperture_flux_errors[i][source_index], flags[i]])
            
            # annulus
            with open(self.cat.out_directory + f"annulus_light_curves/{fltr}_source_{source_index + 1}.csv", "w") as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(['TDB', "flux", "flux_error", "local_background", "local_background_error",
                                    "local_background_per_pixel", "local_background_error_per_pixel", "quality_flag"])
                for i in range(len(self.cat.camera_files[fltr])):
                    csvwriter.writerow([bdts[i], annulus_fluxes[i][source_index], annulus_flux_errors[i][source_index],
                                        local_backgrounds[i][source_index], local_background_errors[i][source_index],
                                        local_backgrounds_per_pixel[i][source_index],
                                        local_background_errors_per_pixel[i][source_index], flags[i]])

    def _parse_results(self, results):
        """
        Parse the photometry results.
        
        Parameters
        ----------
        results :
            The photometry results.
        
        Returns
        -------
        Tuple
            The parsed results.
        """
        
        aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = zip(*results)
        
        return list(aperture_fluxes), list(aperture_flux_errors), list(annulus_fluxes), list(annulus_flux_errors), list(local_backgrounds), list(local_background_errors), list(local_backgrounds_per_pixel), list(local_background_errors_per_pixel), list(flags)
