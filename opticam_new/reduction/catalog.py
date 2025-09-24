from functools import partial
import json
import logging
from multiprocessing import cpu_count
import os
from typing import Callable, Dict, Iterable, List, Literal, Tuple
import warnings

from astroalign import find_transform
from astropy.stats import SigmaClip
from astropy.table import QTable
from astropy.visualization.mpl_normalize import simple_norm
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from photutils.background import Background2D
from photutils.segmentation import SourceCatalog, detect_threshold
from PIL import Image
from skimage.transform import warp, matrix_transform, SimilarityTransform
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from opticam_new.reduction.background import BaseBackground, DefaultBackground
from opticam_new.reduction.correctors import FlatFieldCorrector
from opticam_new.reduction.finder import DefaultFinder
from opticam_new.reduction.photometers import BasePhotometer
from opticam_new.reduction.transforms import find_translation
from opticam_new.utils.batching import get_batches, get_batch_size
from opticam_new.utils.constants import bar_format, pixel_scales
from opticam_new.utils.data_checks import check_data
from opticam_new.utils.helpers import camel_to_snake
from opticam_new.utils.fits_handlers import get_data, save_stacked_images
from opticam_new.utils.logging import recursive_log
from opticam_new.utils.plots import plot_backgrounds, plot_catalogs, plot_time_between_files


class Catalog:
    """
    Create a catalog of sources from OPTICAM data.
    """
    
    def __init__(
        self,
        out_directory: str,
        data_directory: None | str = None,
        c1_directory: None | str = None,
        c2_directory: None | str = None,
        c3_directory: None | str = None,
        rebin_factor: int = 1,
        flat_corrector: None | FlatFieldCorrector = None,
        background: BaseBackground | None = None,
        finder: None | Callable = None,
        threshold: float = 5,
        aperture_selector: Callable = np.median,
        remove_cosmic_rays: bool = True,
        number_of_processors: int = cpu_count() // 2,
        show_plots: bool = True,
        verbose: bool = True
        ) -> None:
        """
        Helper class for reducing OPTICAM data.
        
        Parameters
        ----------
        out_directory: str
            The path to the directory to save the output files.
        data_directory: str, optional
            The path to the directory containing the data, by default `None`. If `None`, any of c1_directory, c2_directory,
            or c3_directory must be defined. If data_directory is defined, c1_directory, c2_directory, and c3_directory
            are ignored.
        c1_directory: str, optional
            The path to the directory containing the C1 data, by default `None`. If `None`, any of data_directory,
            c2_directory, or c3_directory must be defined. This parameter is ignored if data_directory is defined.
        c2_directory: str, optional
            The path to the directory containing the C2 data, by default `None`. If `None`, any of data_directory,
            c1_directory, or c3_directory must be defined. This parameter is ignored if data_directory is defined.
        c3_directory: str, optional
            The path to the directory containing the C3 data, by default `None`. If `None`, any of data_directory,
            c1_directory, or c2_directory must be defined. This parameter is ignored if data_directory is defined.
        rebin_factor: int, optional
            The rebinning factor, by default 1 (no rebinning). The rebinning factor is the factor by which the image is
            rebinned in both dimensions. Rebinning can improve the detectability of faint sources and speed up
            some operations (like cosmic ray removal) at the cost of image resolution.
        flat_corrector: FlatFieldCorrector, optional,
            The flat-field corrector, by default `None`. If `None`, no flat-field corrections are applied.
        threshold: float, optional
            The signal-to-noise ratio threshold for source finding, by default 5. Reduce this value to identify fainter
            sources, though this may lead to the identification of spurious sources.
        background: BaseBackground | None, optional
            The background calculator, by default `None`. If `None`, the default background calculator is used. If a
            callable is provided, it should take an image (`NDArray`) as input and return a `Background2D` object.
        finder: Callable, optional
            The source finder, by default `None`. If `None`, the default source finder is used. If a callable is
            provided, it should take an image (`NDArray`) and a threshold (`float`) as input and return a
            `SegmentationImage` object.
        aperture_selector: Callable, optional
            The aperture selector, by default `np.median`. This function is used to select the aperture size for
            photometry. If a callable is provided, it should take a list of source sizes (`List[float]`) as input and
            return a single value.
        remove_cosmic_rays: bool, optional
            Whether to remove cosmic rays from images, by default True. Cosmic rays are removed using the LACosmic
            algorithm as implemented in `astroscrappy`. Note: this can be computationally expensive, particularly for
            large images (i.e., low binning factors).
        number_of_processors: int, optional
            The number of processors to use for parallel processing, by default half the number of available processors.
        show_plots: bool, optional
            Whether to show plots as they're created, by default `True`. Whether `True` or `False`, plots are always
            saved to `out_directory`.
        verbose: bool, optional
            Whether to print verbose output, by default `True`.
        """
        
        warnings.warn(f'[OPTICAM] from version 0.3.0, opticam_new.Catalog() will be renamed to opticam_new.Reducer()')
        
        self.verbose = verbose
        
        ########################################### out_directory ###########################################
        
        self.out_directory = out_directory
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory):
            if self.verbose:
                print(f"[OPTICAM] {self.out_directory} not found, attempting to create ...")
            # create output directory if it does not exist
            try:
                os.makedirs(self.out_directory)
            except:
                raise FileNotFoundError(f"[OPTICAM] Could not create directory {self.out_directory}")
            if self.verbose:
                print(f"[OPTICAM] {self.out_directory} created.")
        
        ########################################### logger ###########################################
        
        # configure logger
        self.logger = logging.getLogger('OPTICAM')
        self.logger.setLevel(logging.INFO)
        
        # clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # create file handler
        file_handler = logging.FileHandler(os.path.join(self.out_directory, 'info.log'))
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        ########################################### sub-directories ###########################################
        
        # create subdirectories
        if not os.path.isdir(os.path.join(self.out_directory, "cat")):
            os.makedirs(os.path.join(self.out_directory, "cat"))
        if not os.path.isdir(os.path.join(self.out_directory, "diag")):
            os.makedirs(os.path.join(self.out_directory, "diag"))
        if not os.path.isdir(os.path.join(self.out_directory, "misc")):
            os.makedirs(os.path.join(self.out_directory, "misc"))
        
        ########################################### data directories ###########################################
        
        self.data_directory = data_directory
        self.c1_directory = c1_directory
        self.c2_directory = c2_directory
        self.c3_directory = c3_directory
        
        assert self.data_directory is not None or self.c1_directory is not None or self.c2_directory is not None or self.c3_directory is not None, "[OPTICAM] At least one of data_directory, c1_directory, c2_directory, or c3_directory must be defined."
        
        ########################################### input params ###########################################
        
        # set some useful parameters
        self.rebin_factor = rebin_factor
        self.flat_corrector = flat_corrector
        self.aperture_selector = aperture_selector
        self.threshold = threshold
        self.remove_cosmic_rays = remove_cosmic_rays
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        ########################################### check input data ###########################################
        
        self.camera_files, self.binning_scale, self.bmjds, self.ignored_files, self.gains, self.t_ref = check_data(
                data_directory=data_directory,
                c1_directory=c1_directory,
                c2_directory=c2_directory,
                c3_directory=c3_directory,
                out_directory=out_directory,
                verbose=verbose,
                return_output=True,
                logger=self.logger,
                number_of_processors=number_of_processors,
                )  # type: ignore
        
        ########################################### plot time between files ###########################################
        
        plot_time_between_files(
            out_directory=self.out_directory,
            camera_files=self.camera_files,
            bmjds=self.bmjds,
            show=self.show_plots,
            save=True,
            )
        
        ########################################### define reference images ###########################################
        
        # define middle image as reference image for each filter
        reference_indices = {}
        self.reference_files = {}
        for key in self.camera_files.keys():
            reference_indices[key] = len(self.camera_files[key]) // 2
            self.reference_files[key] = self.camera_files[key][reference_indices[key]]
        
        ########################################### aperture selector ###########################################
        
        assert callable(aperture_selector), "[OPTICAM] aperture_selector must be callable."
        self.aperture_selector = aperture_selector
        
        ########################################### background ###########################################
        
        if background is None:
            box_size = 2048 // self.binning_scale // self.rebin_factor // 16
            self.background = DefaultBackground(box_size)
            self.logger.info(f'[OPTICAM] Using default background estimator with box_size={box_size}.')
        elif callable(background):
            # use custom background estimator
            self.background = background
            self.logger.info(f'[OPTICAM] Using custom background estimator {background.__class__.__name__} with parameters {background.__dict__}.')
        else:
            raise ValueError('[OPTICAM] background must be a callable or None. If None, the default background estimator is used.')
        
        ########################################### finder ###########################################
        
        if finder is None:
            effective_image_size = 2048 // self.binning_scale // self.rebin_factor
            npixels = 128 // (2048 // effective_image_size)**2
            border_width = 2048 // self.binning_scale // self.rebin_factor // 16
            self.finder = DefaultFinder(npixels, border_width)
            self.logger.info(f'[OPTICAM] Using default source finder with npixels={npixels} and border_width={border_width}.')
        elif callable(finder):
            self.finder = finder
            self.logger.info(f'[OPTICAM] Using custom source finder {finder.__class__.__name__} with parameters {finder.__dict__}.')
        else:
            raise ValueError('[OPTICAM] finder must be a callable or None. If None, the default source finder is used.')
        
        ########################################### log input params ###########################################
        
        log_reducer_params(self)
        
        ########################################### misc attributes ###########################################
        
        self.transforms = {}  # define transforms as empty dictionary
        self.unaligned_files = []  # define unaligned files as empty list
        self.catalogs = {}  # define catalogs as empty dictionary
        self.psf_params = {}  # define PSF parameters as empty dictionary
        
        ########################################### read transforms ###########################################
        
        if os.path.isfile(os.path.join(self.out_directory, "cat/transforms.json")):
            with open(os.path.join(self.out_directory, "cat/transforms.json"), "r") as file:
                self.transforms.update(json.load(file))
            if self.verbose:
                self.logger.info("[OPTICAM] Read transforms from file.")
        
        ########################################### read catalogs ###########################################
        
        for fltr in list(self.camera_files.keys()):
            if os.path.isfile(os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv")):
                self.catalogs.update(
                    {
                        fltr: QTable.read(
                            os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv"),
                            format="ascii.ecsv",
                            )
                        }
                    )
                self._set_psf_params(fltr)
                if self.verbose:
                    print(f"[OPTICAM] Read {fltr} catalog from file.")

    def _set_psf_params(self, fltr: str) -> None:
        """
        Set the PSF parameters for a given filter based on the catalog data.
        
        Parameters
        ----------
        fltr : str
            The filter for which to set the PSF parameters.
        """
        
        self.psf_params[fltr] = {
            'semimajor_sigma': self.aperture_selector(self.catalogs[fltr]['semimajor_sigma'].value),
            'semiminor_sigma': self.aperture_selector(self.catalogs[fltr]['semiminor_sigma'].value),
            'orientation': self.aperture_selector(self.catalogs[fltr]['orientation'].value)
        }
        
        self.logger.info(f'[OPTICAM] {fltr} PSF parameters:')
        self.logger.info(f'[OPTICAM]    semimajor_sigma: {self.psf_params[fltr]["semimajor_sigma"]} binned pixels ({self.psf_params[fltr]["semimajor_sigma"] * self.binning_scale * self.rebin_factor * pixel_scales[fltr]} arcsec)')
        self.logger.info(f'[OPTICAM]    semiminor_sigma: {self.psf_params[fltr]["semiminor_sigma"]} binned pixels ({self.psf_params[fltr]["semiminor_sigma"] * self.binning_scale * self.rebin_factor * pixel_scales[fltr]} arcsec)')
        self.logger.info(f'[OPTICAM]    orientation: {self.psf_params[fltr]["orientation"]} degrees')

    def _get_source_coords_from_image(
        self,
        image: NDArray,
        bkg: Background2D | None = None,
        away_from_edge: bool | None = False,
        n_sources: int | None = None,
        ) -> NDArray:
        """
        Get an array of source coordinates from an image in descending order of source brightness.
        
        Parameters
        ----------
        image : NDArray
            The **non-background-subtracted** image from which to extract source coordinates.
        bkg : Background2D, optional
            The background of the image, by default None. If None, the background is estimated from the image.
        away_from_edge : bool, optional
            Whether to exclude sources near the edge of the image, by default False.
        n_sources : int, optional
            The number of source coordinates to return, by default `None` (all sources will be returned).
        
        Returns
        -------
        NDArray
            The source coordinates in descending order of brightness.
        """
        
        if bkg is None:
            bkg = self.background(image)  # get background
        
        image_clean = image - bkg.background  # remove background from image
        
        cat = self.finder(image_clean, self.threshold*bkg.background_rms)  # find sources in background-subtracted image
        tbl = SourceCatalog(image_clean, cat, background=bkg.background).to_table()  # create catalog of sources
        tbl.sort('segment_flux', reverse=True)  # sort catalog by flux in descending order
        
        coords = np.array([tbl["xcentroid"], tbl["ycentroid"]]).T
        
        if away_from_edge:
            edge = self.background.box_size
            for coord in coords:
                if coord[0] < edge or coord[0] > image.shape[1] - edge or coord[1] < edge or coord[1] > image.shape[0] - edge:
                    coords = np.delete(coords, np.where(np.all(coords == coord, axis=1)), axis=0)
        
        if n_sources is not None:
            coords = coords[:n_sources]
        
        return coords


    def create_catalogs(
        self,
        max_catalog_sources: int = 50,
        n_alignment_sources: int = 3,
        transform_type: Literal['affine', 'translation'] = 'translation',
        rotation_limit: float | None = None,
        translation_limit: float | int | List[float | int] | None = None,
        scale_limit: float | None = None,
        overwrite: bool = False,
        show_diagnostic_plots: bool = False,
        ) -> None:
        """
        Initialise the source catalogs for each camera. Some aspects of this method are parallelised for speed.
        
        Parameters
        ----------
        max_catalog_sources : int, optional
            The maximum number of sources above the specified threshold that will be included in the catalog, by default
            50. Only the brightest max_catalog_sources sources are included in the catalog.
        n_alignment_sources : int, optional
            The (maximum) number of sources to use for image alignment, by default 3. If `transform_type='translation'`,
            `n_alignment_sources` must be >= 1, and the brightest `n_alignment_sources` sources are used for image
            alignment. If `transform_type='affine'`, `n_alignment_sources` must be >= 3 and represents that *maximum*
            number of sources that *may* be used for image alignment.
        transform_type : Literal['affine', 'translation'], optional
            The type of transform to use for image alignment, by default 'translation'. 'translation' performs simple
            x, y translations, while 'affine' uses `astroalign.find_transform()`.
        rotation_limit : float, optional
            The maximum rotation limit (in degrees) for affine transformations, by default `None` (no limit).
        scale_limit : float, optional
            The maximum scale limit for affine transformations, by default `None` (no limit).
        translation_limit : float | int | List[float | int] | None, optional
            The maximum translation limit for transformations, by default `None` (no limit). Can be a scalar value that
            applies to both x- and y-translations, or an iterable where the first value defines the x-translation limit
            and the second value defines the y-translation limit.
        overwrite : bool, optional
            Whether to overwrite existing catalogs, by default False.
        show_diagnostic_plots : bool, optional
            Whether to show diagnostic plots, by default False. Diagnostic plots are saved to out_directory, so this
            parameter only affects whether the plots are displayed in the console.
        """
        
        assert transform_type in ['affine', 'translation'], '[OPTICAM] transform_type must be either "affine" or "translation".'
        
        if transform_type=='translation':
            warnings.warn(f'[OPTICAM] transform_type="translation" will be deprecated in version 0.3.0, and the default will be transform_type="affine".')
        
        if translation_limit is not None:
            # if a scalar translation limit is specified, convert it to a list
            if isinstance(translation_limit, float) or isinstance(translation_limit, int):
                translation_limit = [translation_limit, translation_limit]
        
        # if catalogs already exist, skip
        if os.path.isfile(os.path.join(self.out_directory, 'cat/catalogs.png')) and not overwrite:
            print('[OPTICAM] Catalogs already exist. To overwrite, set overwrite to True.')
            return
        
        if self.verbose:
            print('[OPTICAM] Initialising catalogs')
        
        background_median = {}
        background_rms = {}
        stacked_images = {}
        
        # for each camera
        for fltr in self.camera_files.keys():
            
            # if no images found for camera, skip
            if len(self.camera_files[fltr]) == 0:
                continue
            
            # get reference image
            # np.asarray() to fix type error
            reference_image = np.asarray(
                get_data(
                    file=self.reference_files[fltr],
                    gain=self.gains[self.reference_files[fltr]],
                    flat_corrector=self.flat_corrector,
                    rebin_factor=self.rebin_factor,
                    return_error=False,
                    remove_cosmic_rays=self.remove_cosmic_rays,
                    )
                )
            
            try:
                reference_coords = self._get_source_coords_from_image(reference_image, away_from_edge=True, n_sources=n_alignment_sources)  # get source coordinates in descending order of brightness
            except:
                self.logger.info(f'[OPTICAM] No sources detected in {fltr} reference image ({self.reference_files[fltr]}). Reducing threshold or npixels in the source finder may help.')
                continue
            
            if len(reference_coords) < n_alignment_sources:
                self.logger.info(f'[OPTICAM] Found {len(reference_coords)} sources in {fltr} reference image ({self.reference_files[fltr]}) but n_alignment_sources={n_alignment_sources}. Consider reducing threshold and/or n_alignment_sources.')
                continue
            
            self.logger.info(f'[OPTICAM] {fltr} alignment source coordinates: {reference_coords}')
            
            # align and stack images in batches
            batches = get_batches(self.camera_files[fltr])
            results = process_map(
                partial(
                    self._align_images,
                    reference_image_shape=reference_image.shape,
                    reference_coords=reference_coords,
                    transform_type=transform_type,
                    rotation_limit=rotation_limit,
                    scale_limit=scale_limit,
                    translation_limit=translation_limit,
                    n_alignment_sources=n_alignment_sources,
                    ),
                batches,
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f'[OPTICAM] Aligning {fltr} images',
                bar_format=bar_format,
                tqdm_class=tqdm,
                )
            stacked_image, background_medians, background_rmss = self._parse_alignment_results(
                results,
                fltr,
                )
            
            background_median[fltr] = background_medians
            background_rms[fltr] = background_rmss
            
            try:
                # estimate threshold for source detection
                threshold = detect_threshold(
                    stacked_image,
                    nsigma=self.threshold,
                    sigma_clip=SigmaClip(sigma=3, maxiters=10),
                    )
            except:
                self.logger.info('[OPTICAM] Unable to estimate source detection threshold for ' + fltr + ' stacked image.')
                continue
            
            try:
                # identify sources in stacked image
                segment_map = self.finder(
                    stacked_image,
                    threshold,
                    )
            except:
                self.logger.info('[OPTICAM] No sources detected in the stacked ' + fltr + ' stacked image. Reducing threshold may help.')
                continue
            
            # save stacked image and its background
            stacked_images[fltr] = stacked_image
            
            tbl = SourceCatalog(stacked_image, segment_map).to_table()  # create catalog of sources
            tbl.sort('segment_flux', reverse=True)
            tbl = tbl[:max_catalog_sources]  # limit catalog to brightest max_catalog_sources sources
            
            # create catalog of sources in stacked image and write to file
            self.catalogs.update({fltr: tbl})
            self.catalogs[fltr].write(
                os.path.join(self.out_directory, f"cat/{fltr}_catalog.ecsv"),
                format="ascii.ecsv",
                overwrite=True,
                )
            
            self._set_psf_params(fltr)  # set PSF parameters for the filter
        
        save_stacked_images(
            stacked_images=stacked_images,
            out_directory=self.out_directory,
            overwrite=overwrite,
            )
        
        plot_catalogs(
            out_directory=self.out_directory,
            filters=list(self.camera_files.keys()),
            stacked_images=stacked_images,
            catalogs=self.catalogs,
            show=self.show_plots,
            save=True,
        )
        
        plot_backgrounds(
            out_directory=self.out_directory,
            camera_files=self.camera_files,
            background_median=background_median,
            background_rms=background_rms,
            bmjds=self.bmjds,
            t_ref=self.t_ref,
            show=show_diagnostic_plots,
            save=True,
        )
        
        # diagnostic plots
        # self._plot_background_meshes(stacked_images, show_diagnostic_plots)  # plot background meshes TODO: make dedicated routine for pre-reduction sanity checking?
        # TODO: make below dedicated routines for post-reduction analysis?
        # for (fltr, stacked_image) in stacked_images.items():
        #     self._visualise_psfs(stacked_image, fltr, show_diagnostic_plots)
        
        # save transforms to file
        with open(os.path.join(self.out_directory, "cat/transforms.json"), "w") as file:
            json.dump(self.transforms, file, indent=4)
        
        # write unaligned files to file
        if len(self.unaligned_files) > 0:
            with open(os.path.join(self.out_directory, "diag/unaligned_files.txt"), "w") as unaligned_file:
                for file in self.unaligned_files:
                    unaligned_file.write(file + "\n")

    def _align_images(
        self,
        batch: List[str],
        reference_image_shape: Tuple[int],
        reference_coords: NDArray,
        transform_type: Literal['affine', 'translation'],
        rotation_limit: float | None,
        scale_limit: float | None,
        translation_limit: List[float] | None,
        n_alignment_sources: int,
        ) -> Tuple[
            NDArray,
            Dict[str, float],
            Dict[str, float],
            Dict[str, float],
            ]:
        """
        Align an image based on some reference coordinates.
        
        Parameters
        ----------
        file: str
            The file path.
        reference_image : NDArray
            The reference image.
        reference_coords : NDArray
            The source coordinates in the reference image.
        transform_type : Literal['affine', 'translation']
            The type of transform to use for image alignment.
        rotation_limit : float | None
            The maximum rotation limit (in degrees) for image alignment.
        scale_limit : float | None
            The maximum scaling limit for image alignment.
        translation_limit : List[float] | None
            The maximum translation limit for image alignment.
        n_alignment_sources : int
            The (maximum) number of sources to use for image alignment.
        
        Returns
        -------
        Tuple[List[float], float, float]
            The transform parameters, background median, and background RMS.
        """
        
        stacked_image = np.zeros(reference_image_shape)  # create empty stacked image
        transforms = {}
        background_medians = {}
        background_rmss = {}
        
        for file in batch:
            
            data = np.asarray(
                get_data(
                    file,
                    self.gains[file],
                    flat_corrector=self.flat_corrector,
                    rebin_factor=self.rebin_factor,
                    return_error=False,
                    remove_cosmic_rays=self.remove_cosmic_rays),
                )
            
            # calculate and subtract background
            bkg = self.background(data)
            background_median = bkg.background_median
            background_rms = bkg.background_rms_median
            
            # identify sources
            try:
                coords = self._get_source_coords_from_image(data, bkg)
            except:
                self.logger.info(f'[OPTICAM] No sources detected in {file}.')
                continue
            
            if len(coords) < len(reference_coords):
                self.logger.info(f'[OPTICAM] n_alignment_sources={len(reference_coords)} but only {len(reference_coords)} sources detected in {file}. Skipping.')
                continue
            
            if transform_type == 'translation':
                # find translation
                transform = find_translation(
                    coords,
                    reference_coords,
                    )
            else:
                # find affine transformation using astroalign
                try:
                    transform = find_transform(
                        coords,
                        reference_coords,
                        max_control_points=n_alignment_sources,
                        )[0]
                except Exception as e:
                    self.logger.info(f'[OPTICAM] Could not align {file} due to the following exception: {e}. Skipping.')
                    continue
            
            # validate transform
            if not self._valid_transform(
                file=file,
                transform=transform,
                rotation_limit=rotation_limit,
                scale_limit=scale_limit,
                translation_limit=translation_limit):
                continue
            
            transforms[file] = transform.params.tolist()  # type: ignore
            background_medians[file] = background_median
            background_rmss[file] = background_rms
            
            # transform and stack image
            stacked_image += warp(
                data - bkg.background,
                transform.inverse,
                output_shape=reference_image_shape,
                order=3,
                mode='constant',
                cval=float(np.nanmedian(data)),
                clip=True,
                preserve_range=True,
                )
        
        return stacked_image, transforms, background_medians, background_rmss

    def _valid_transform(
        self,
        file: str,
        transform: SimilarityTransform,
        rotation_limit: float | None,
        scale_limit: float | None,
        translation_limit: List[float] | None,
        ) -> bool:
        """
        Find whether a transform is valid given some transform limits.
        
        Parameters
        ----------
        file : str
            The file being transformed.
        transform : SimilarityTransform
            The transform.
        rotation_limit : float | None
            The rotation limit.
        scale_limit : float | None
            The scale limit.
        translation_limit : List[float] | None
            The translation limit.
        
        Returns
        -------
        bool
            Whether the transform is valid.
        """
        
        if rotation_limit:
            if abs(transform.rotation) > rotation_limit:
                self.logger.info(f'[OPTICAM] File {file} transform exceeded rotation limit. Rotation limit is {rotation_limit}, but rotation was {transform.rotation}.')
                return False
        if scale_limit:
            if transform.scale > scale_limit:
                self.logger.info(f'[OPTICAM] File {file} transform exceeded scale limit. Scale limit is {scale_limit}, but scale was {transform.scale}.')
                return False
        if translation_limit:
            if abs(transform.translation[0]) > translation_limit[0] or abs(transform.translation[1]) > translation_limit[1]:
                self.logger.info(f'[OPTICAM] File {file} transform exceeded translation limit. Translation limit is {translation_limit}, but translation was {transform.translation}.')
                return False
        
        return True

    def _parse_alignment_results(
        self,
        results: Tuple,
        fltr: str,
        ) -> Tuple[NDArray, Dict[str, float], Dict[str, float]]:
        """
        Parse the results of image alignment.
        
        Parameters
        ----------
        results : Tuple
            The results.
        fltr : str
            The filter.
        
        Returns
        -------
        Tuple[NDArray, Dict[str, float], Dict[str, float]]:
            The stacked image, background medians, and background RMSs.
        """
        
        transforms = {}
        unaligned_files = []
        background_medians = {}
        background_rmss = {}
        
        # unpack results
        batch_stacked_images, batch_transforms, batch_background_medians, batch_background_rmss = zip(*results)
        
        # combine results
        for i in range(len(batch_stacked_images)):
            transforms.update(batch_transforms[i])
            background_medians.update(batch_background_medians[i])
            background_rmss.update(batch_background_rmss[i])
        
        aligned_files = list(transforms.keys())
        for file in self.camera_files[fltr]:
            if file not in aligned_files:
                unaligned_files.append(file)
        
        stacked_image = np.sum(batch_stacked_images, axis=0)  # stack images
        
        self.transforms.update(transforms)  # update transforms
        self.unaligned_files += unaligned_files  # update unaligned files
        
        if self.verbose:
            print(f"[OPTICAM] Done.")
            print(f'[OPTICAM] {len(transforms)} image(s) aligned.')
            print(f'[OPTICAM] {len(unaligned_files)} image(s) could not be aligned.')
        
        return stacked_image, background_medians, background_rmss


    def _plot_background_meshes(self, stacked_images: Dict[str, NDArray], show: bool) -> None:
        """
        Plot the background meshes on top of the catalog images.
        
        Parameters
        ----------
        stacked_images : Dict[str, NDArray]
            The stacked images for each camera.
        show : bool
            Whether to display the plot.
        """
        
        fig, ax = plt.subplots(ncols=len(self.catalogs), tight_layout=True, figsize=(len(self.catalogs) * 5, 5))
        
        for i, fltr in enumerate(list(self.catalogs.keys())):
            
            plot_image = np.clip(stacked_images[fltr], 0, None)
            bkg = self.background(stacked_images[fltr])
            
            try:
                # plot background mesh
                ax[i].imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                             norm=simple_norm(plot_image, stretch="log"))
                bkg.plot_meshes(ax=ax[i], outlines=True, marker='.', color='cyan', alpha=0.3)
                
                #label plot
                ax[i].set_title(fltr)
                ax[i].set_xlabel("X")
                ax[i].set_ylabel("Y")
            except:
                # plot background mesh
                ax.imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                          norm=simple_norm(plot_image, stretch="log"))
                bkg.plot_meshes(ax=ax, outlines=True, marker='.', color='cyan', alpha=0.3)
                
                # label plot
                ax.set_title(fltr)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

        fig.savefig(os.path.join(self.out_directory, "diag/background_meshes.png"))

        if show and self.show_plots:
            plt.show(fig)
        else:
            fig.clear()
            plt.close(fig)

    def _visualise_psfs(self, image: NDArray, fltr: str, show: bool) -> None:
        """
        Generate PSF plots for each source in an image.
        
        Parameters
        ----------
        image : NDArray
            The image (not background subtracted).
        fltr : str
            The image filter.
        show: bool
            Whether to display the plot.
        """
        
        # estimate source threshold
        try:
            threshold = detect_threshold(image, nsigma=self.threshold, sigma_clip=SigmaClip(sigma=3, maxiters=10))
        except:
            print('[OPTICAM] Unable to estimate source detection threshold when trying to visualise source PSFs.')
            return
        
        # find sources in image
        try:
            segm = self.finder(image, threshold) 
        except:
            print('[OPTICAM] No sources detected when trying to visualise source PSFs.')
            return
        
        # get source table
        tbl = SourceCatalog(image, segm).to_table()
        
        x_lo, x_hi = 0, image.shape[1]
        y_lo, y_hi = 0, image.shape[0]
        
        for source in tbl['label']:
            x, y = int(tbl['xcentroid'][source - 1]), int(tbl['ycentroid'][source - 1])  # source position
            w = int(tbl['semimajor_sigma'][source - 1].value) * 5  # source width
            x_range = np.arange(min(x_lo, int(x - w)), max(x_hi, int(x + w)))  # x range
            y_range = np.arange(min(y_lo, int(y - w)), max(y_hi, int(y + w)))  # y range
            
            # create mask
            mask = np.zeros_like(image, dtype=bool)
            for x in x_range:
                for y in y_range:
                    mask[y, x] = True
            
            # isolate source
            rows_to_keep = np.any(mask, axis=1)
            star_data = image[rows_to_keep, :]
            cols_to_keep = np.any(mask, axis=0)
            star_data = star_data[:, cols_to_keep]
            
            fig = plt.figure(num=1, clear=True)
            ax = fig.add_subplot(projection='3d')
            
            x, y = np.meshgrid(x_range, y_range)
            
            ax.plot_surface(x, y, star_data, edgecolor='r', rstride=2, cstride=2, color='none', lw=.5)
            ax.contour(x, y, star_data, 20, zdir='x', offset=ax.set_xlim()[0], colors='black', linewidths=.5)
            ax.contour(x, y, star_data, 20, zdir='y', offset=ax.set_ylim()[1], colors='black', linewidths=.5)
            ax.contour(x, y, star_data, 10, zdir='z', offset=ax.set_zlim()[0], colors='black', linewidths=.5)
            
            ax.view_init(30, -45, 0)
            
            ax.set_title(f'{fltr} source {source}')
            
            fig.savefig(os.path.join(self.out_directory, f'diag/{fltr}_source_{source}_psf.png'))
            
            if show and self.show_plots:
                plt.show()
            else:
                fig.clear()
                plt.close(fig)


    def create_gifs(self, keep_frames: bool = True, overwrite: bool = False) -> None:
        """
        Create alignment gifs for each camera. Some aspects of this method are parallelised for speed. The frames are 
        saved in out_directory/diag/*-band_gif_frames and the GIFs are saved in out_directory/cat.
        
        Parameters
        ----------
        keep_frames : bool, optional
            Whether to save the GIF frames in out_directory/diag, by default True. If False, the frames will be deleted
            after the GIF is saved.
        overwrite : bool, optional
            Whether to overwrite existing GIFs, by default False.
        """
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            elif os.path.exists(os.path.join(self.out_directory, f"cat/{fltr}_images.gif")) and not overwrite:
                print(f"[OPTICAM] {fltr} GIF already exists. To overwrite, set overwrite to True.")
                continue
            
            # create gif frames directory if it does not exist
            if not os.path.isdir(os.path.join(self.out_directory, f"diag/{fltr}_gif_frames")):
                os.mkdir(os.path.join(self.out_directory, f"diag/{fltr}_gif_frames"))
            
            chunksize = max(1, len(self.camera_files[fltr]) // 100)  # chunk size for parallel processing (must be >= 1)
            process_map(partial(self._create_gif_frames, fltr=fltr), self.camera_files[fltr],
                        max_workers=self.number_of_processors, disable=not self.verbose,
                        desc=f"[OPTICAM] Creating {fltr} GIF frames", chunksize=chunksize)
            
            # save GIF
            self._compile_gif(fltr, keep_frames)

    def _create_gif_frames(self, file: str, fltr: str) -> None:
        """
        Create a gif frames from a batch of images and save it to the out_directory.
        
        Parameters
        ----------
        file : str
            The list of file names in the batch.
        fltr : str
            The filter.
        """
        
        data = get_data(
            file=file,
            gain=self.gains[file],
            flat_corrector=self.flat_corrector,
            rebin_factor=self.rebin_factor,
            return_error=False,
            remove_cosmic_rays=self.remove_cosmic_rays,
            )
        
        file_name = file.split('/')[-1].split(".")[0]
        
        bkg = self.background(data)
        clean_data = data - bkg.background
        
        # clip negative values to zero for better visualisation
        plot_image = np.clip(clean_data, 0, None)
        
        fig, ax = plt.subplots(num=1, clear=True, tight_layout=True)
        
        ax.imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                norm=simple_norm(plot_image, stretch="log"))
        
        # for each source
        for i in range(len(self.catalogs[fltr])):
            
            source_position = (self.catalogs[fltr]["xcentroid"][i], self.catalogs[fltr]["ycentroid"][i])
            
            if file == self.reference_files[fltr]:
                aperture_position = source_position
                title = f"{file_name} (reference)"
                colour = "blue"
            else:
                try:
                    aperture_position = matrix_transform(source_position, self.transforms[file])[0]
                    title = f"{file_name} (aligned)"
                    colour = "black"
                except:
                    aperture_position = source_position
                    title = f"{file_name} (unaligned)"
                    colour = "red"
            
            radius = 5 * self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            
            ax.add_patch(Circle(xy=(aperture_position), radius=radius,
                                    edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1))
            # ax.add_patch(Circle(xy=(aperture_position), radius=self.local_background.r_in_scale*radius,
            #                     edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1, ls=":"))
            # ax.add_patch(Circle(xy=(aperture_position), radius=self.local_background.r_out_scale*radius,
            #                     edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1, ls=":"))
            ax.text(aperture_position[0] + 1.05*radius, aperture_position[1] + 1.05*radius, i + 1,
                        color=self.colours[i % len(self.colours)])
        
        ax.set_title(title, color=colour)
        
        fig.savefig(os.path.join(self.out_directory, f'diag/{fltr}_gif_frames/{file_name}.png'))

    def _compile_gif(self, fltr: str, keep_frames: bool) -> None:
        """
        Create a gif from the frames saved in out_directory.

        Parameters
        ----------
        fltr : str
            The filter.
        keep_frames : bool
            Whether to keep the frames after the gif is saved.
        """
        
        # load frames
        frames = []
        for file in tqdm(self.camera_files[fltr], disable=not self.verbose, desc=f"[OPTICAM] Loading {fltr} GIF frames"):
            try:
                frames.append(Image.open(os.path.join(self.out_directory, f'diag/{fltr}_gif_frames/{file.split('/')[-1].split(".")[0]}.png')))
            except:
                pass
        
        # save gif
        frames[0].save(
            os.path.join(
                self.out_directory,
                f'cat/{fltr}_images.gif',
                ),
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=200,
            loop=0,
            )
        
        # close images
        for frame in frames:
            frame.close()
        del frames
        
        # delete frames after gif is saved
        if not keep_frames:
            for file in tqdm(os.listdir(os.path.join(self.out_directory, f"diag/{fltr}_gif_frames")), disable=not self.verbose,
                             desc=f"[OPTICAM] Deleting {fltr} GIF frames"):
                os.remove(os.path.join(self.out_directory, f"diag/{fltr}_gif_frames/{file}"))


    def photometry(
        self,
        photometer: BasePhotometer,
        ) -> None:
        """
        Perform photometry on the catalogs using the provided photometer.
        
        Parameters
        ----------
        photometer : BasePhotometer
            The photometer. Should be a subclass of `BasePhotometer`, or implement a `compute` method that follows the
            `BasePhotometer` interface.
        """
        
        # define save directory using the photometer name
        save_name = camel_to_snake(photometer.__class__.__name__).replace('_photometer', '')
        
        # change save directory based on photometer settings
        if photometer.local_background_estimator is not None:
            save_name += '_annulus'
        if not photometer.match_sources:
            save_name = 'forced_' + save_name
        
        print(f'[OPTICAM] Photometry results will be saved to {save_name}_light_curves in {self.out_directory}.')
        
        save_dir = os.path.join(self.out_directory, f"{save_name}_light_curves")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # for each filter
        for fltr in self.catalogs.keys():
            
            source_coords = np.array([self.catalogs[fltr]["xcentroid"].value,
                                      self.catalogs[fltr]["ycentroid"].value]).T
            
            batch_size = get_batch_size(len(self.camera_files[fltr]))
            results = process_map(
                partial(
                    self._photometry,
                    photometer,
                    source_coords,
                    fltr,
                ),
                self.camera_files[fltr],
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f"[OPTICAM] Performing photometry on {fltr} images",
                chunksize=batch_size,
                bar_format=bar_format,
                tqdm_class=tqdm,
            )
            
            # merge multiprocessing results
            photometry_results = {}
            for result in results:
                for key, value in result.items():
                    if key not in photometry_results:
                        photometry_results[key] = []
                    photometry_results[key].append(value)
            
            # for each source in the catalog
            for i in range(len(self.catalogs[fltr])):
                
                # unpack results for ith source
                source_results = {}
                for key, values in photometry_results.items():
                    
                    # time is a special case since it is already a single column
                    if key == 'BMJD':
                        source_results[key] = np.asarray(values)
                    # for other keys, the ith column needs to be extracted
                    else:
                        col = [value[i] for value in values]
                        source_results[key] = np.asarray(col)
                
                # define light curve as a DataFrame
                df = pd.DataFrame(source_results)
                
                # drop NaNs
                df.dropna(inplace=True, ignore_index=True)
                df.reset_index(drop=True, inplace=True)
                
                # save to file
                df.to_csv(
                    os.path.join(
                        save_dir,
                        f'{fltr}_source_{i + 1}.csv',
                        ),
                    index=False,
                    )

    def _photometry(
        self,
        photometer: BasePhotometer,
        source_coords: NDArray,
        fltr: str,
        file: str,
        ) -> Dict[str, List]:
        
        image, error = get_data(
            file=file,
            gain=self.gains[file],
            flat_corrector=self.flat_corrector,
            rebin_factor=self.rebin_factor,
            return_error=True,
            remove_cosmic_rays=self.remove_cosmic_rays,
            )
        
        if photometer.local_background_estimator is None:
            bkg = self.background(image)  # get 2D background
            image = image - bkg.background  # remove background from image
            error = np.sqrt(error**2 + bkg.background_rms**2)  # propagate error
            threshold = self.threshold * bkg.background_rms  # define source detection threshold
        else:
            # estimate source detection threshold from noisy image
            threshold = detect_threshold(image, self.threshold, error=error)
        
        image_coords = None  # assume no image coordinates by default
        if photometer.match_sources:
            try:
                segm = self.finder(image, threshold)
                tbl = SourceCatalog(image, segm).to_table()
                image_coords = np.array([tbl["xcentroid"].value,
                                        tbl["ycentroid"].value]).T
            except Exception as e:
                self.logger.warning(f"[OPTICAM] Could not determine source coordinates in {file}: {e}")
        
        results = photometer.compute(image, error, source_coords, image_coords, self.psf_params[fltr])
        
        assert 'flux' in results, f"[OPTICAM] Photometer {photometer.__class__.__name__}'s compute method must return a 'flux' key."
        assert 'flux_err' in results, f"[OPTICAM] Photometer {photometer.__class__.__name__}'s compute method must return a 'flux_err' key."
        
        # results check
        for key, values in results.items():
            for i, value in enumerate(values):
                if value is None:
                    self.logger.warning(f"[OPTICAM] {key} could not be determined for source {i + 1} in {fltr} (got value {value}).")
        
        # add time stamp
        results['BMJD'] = self.bmjds[file]  # add time of observation
        
        return results





def log_reducer_params(
    reducer: Catalog,
    ) -> None:
    
    # get parameters
    params = dict(recursive_log(reducer, max_depth=5))
    
    params.update({'filters': list(reducer.camera_files.keys())})
    
    # remove some parameters that are either already saved elsewhere or are not relevant
    params.pop('logger')
    params.pop('bmjds')
    params.pop('gains')
    params.pop('camera_files')
    
    try:
        params.pop('transforms')
    except KeyError:
        pass
    
    try:
        params.pop('unaligned_files')
    except KeyError:
        pass
    
    try:
        params.pop('catalogs')
    except KeyError:
        pass
    
    # sort parameters
    params = dict(sorted(params.items()))
    
    # write parameters to file
    with open(os.path.join(reducer.out_directory, "misc/reduction_parameters.json"), "w") as file:
        json.dump(params, file, indent=4)










