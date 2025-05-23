import os

try:
    os.environ['OMP_NUM_THREADS'] = '1'  # set number of threads to 1 for better multiprocessing performance
except:
    pass

from tqdm.contrib.concurrent import process_map  # process_map removes a lot of the boilerplate from multiprocessing
from tqdm import tqdm
from astropy.table import QTable
import json
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.visualization.mpl_normalize import simple_norm
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.segmentation import SourceCatalog, detect_threshold
from photutils.aperture import aperture_photometry, CircularAperture, EllipticalAperture
from photutils.background import Background2D
from photutils.utils import calc_total_error
from skimage.transform import estimate_transform, warp, matrix_transform, SimilarityTransform
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from multiprocessing import cpu_count, Pool
from functools import partial
from PIL import Image
from typing import Any, List, Dict, Literal, Callable, Tuple, Union
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import csv
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ccdproc import cosmicray_lacosmic  # replace with astroscrappy to reduce dependencies?
import logging
from types import FunctionType

from opticam_new.helpers import log_binnings, log_filters, default_aperture_selector, apply_barycentric_correction, clip_extended_sources, rebin_image, get_time
from opticam_new.background import Background
from opticam_new.local_background import EllipticalLocalBackground
from opticam_new.finder import CrowdedFinder, Finder
from opticam_new.correctors import FlatFieldCorrector



# TODO: add FWHM column to catalog tables?


class Reducer:
    """
    Helper class for reducing OPTICAM data.
    """
    
    def __init__(
        self,
        out_directory: str,
        data_directory: str = None,
        c1_directory: str = None,
        c2_directory: str = None,
        c3_directory: str = None,
        rebin_factor: int = 1,
        flat_corrector: FlatFieldCorrector = None,
        threshold: float = 5,
        background: Callable = None,
        local_background: Callable = None,
        finder: Union[Literal['crowded', 'default'], Callable] = 'default',
        aperture_selector: Callable = None,
        scale: float = 5,
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
            The path to the directory containing the data, by default None. If None, any of c1_directory, c2_directory,
            or c3_directory must be defined. If data_directory is defined, c1_directory, c2_directory, and c3_directory
            are ignored.
        c1_directory: str, optional
            The path to the directory containing the C1 data, by default None. If None, any of data_directory,
            c2_directory, or c3_directory must be defined. This parameter is ignored if data_directory is defined.
        c2_directory: str, optional
            The path to the directory containing the C2 data, by default None. If None, any of data_directory,
            c1_directory, or c3_directory must be defined. This parameter is ignored if data_directory is defined.
        c3_directory: str, optional
            The path to the directory containing the C3 data, by default None. If None, any of data_directory,
            c1_directory, or c2_directory must be defined. This parameter is ignored if data_directory is defined.
        rebin_factor: int, optional
            The rebinning factor, by default 1 (no rebinning). The rebinning factor is the factor by which the image is
            rebinned in both dimensions (i.e., a rebin_factor of 2 will reduce the image size by a factor of 4).
            Rebinning can improve the detectability of faint sources.
        flat_corrector: FlatFieldCorrector, optional,
            The flat-field corrector, by default None. If None, no flat-field corrections are applied.
        threshold: float, optional
            The threshold for source finding, by default 5. The threshold is the background RMS factor above which
            sources are detected. For faint sources, a lower threshold may be required.
        background: Callable, optional
            The background calculator, by default None. If None, the default background calculator is used.
        local_background: Callable, optional
            The local background estimator, by default None. If None, the default local background estimator is used.
        finder: Union[Literal['crowded', 'default'], Callable], optional
            The source finder, by default 'default'. If 'default', the default source finder (no deblending) is used.
            If 'crowded', the crowded source finder (with deblending) is used. Alternatively, a custom source finder can
            be provided.
        aperture_selector: Callable, optional
            The aperture selector, by default None. If None, the default aperture selector is used.
        scale: float, optional
            The aperture scale factor, by default 5. The aperture scale factor scales the aperture size returned by
            aperture_selector for forced photometry.
        remove_cosmic_rays: bool, optional
            Whether to remove cosmic rays from images, by default True. Cosmic rays are removed using the LACosmic
            algorithm as implemented in astroscrappy.
        number_of_processors: int, optional
            The number of processors to use for parallel processing, by default half the number of available processors.
            Note that there is some overhead incurred when using multiple processors, so there can be diminishing
            returns when using more processors.
        show_plots: bool, optional
            Whether to show plots as they're created, by default True. Whether True or False, plots are always saved
            to out_directory.
        verbose: bool, optional
            Whether to print verbose output, by default True.
        
        Raises
        ------
        FileNotFoundError
            If the data directory does not exist.
        ValueError
            If the image binning is not "8x8", "4x4", "3x3", "2x2", or "1x1" and a background estimator is not
            provided.
        ValueError
            If the image binning is not "8x8", "4x4", "3x3", "2x2", or "1x1" and a source finder is not provided.
        """
        
        self.verbose = verbose
        
        self.out_directory = out_directory
        if not self.out_directory.endswith("/"):
            self.out_directory += "/"
        
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
        
        # configure logger
        self.logger = logging.getLogger('OPTICAM')
        self.logger.setLevel(logging.INFO)
        
        # clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # create console handler
        file_handler = logging.FileHandler(self.out_directory + 'info.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # create subdirectories
        if not os.path.isdir(self.out_directory + "cat"):
            os.makedirs(self.out_directory + "cat")
        if not os.path.isdir(self.out_directory + "diag"):
            os.makedirs(self.out_directory + "diag")
        if not os.path.isdir(self.out_directory + "misc"):
            os.makedirs(self.out_directory + "misc")
        
        self.data_directory = data_directory
        self.c1_directory = c1_directory
        self.c2_directory = c2_directory
        self.c3_directory = c3_directory
        
        assert self.data_directory is not None or self.c1_directory is not None or self.c2_directory is not None or self.c3_directory is not None, "[OPTICAM] At least one of data_directory, c1_directory, c2_directory, or c3_directory must be defined."
        
        if self.data_directory is not None:
            if not self.data_directory[-1].endswith("/"):
                self.data_directory += "/"
        else:
            if self.c1_directory is not None:
                if not self.c1_directory[-1].endswith("/"):
                    self.c1_directory += "/"
            if self.c2_directory is not None:
                if not self.c2_directory[-1].endswith("/"):
                    self.c2_directory += "/"
            if self.c3_directory is not None:
                if not self.c3_directory[-1].endswith("/"):
                    self.c3_directory += "/"
        
        # set parameters
        self.rebin_factor = rebin_factor
        self.flat_corrector = flat_corrector
        self.fwhm_scale = 2 * np.sqrt(2 * np.log(2))  # FWHM scale factor
        self.aperture_selector = default_aperture_selector if aperture_selector is None else aperture_selector
        self.scale = scale
        self.threshold = threshold
        self.remove_cosmic_rays = remove_cosmic_rays
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        # define file paths
        self.file_paths = []
        if self.data_directory is not None:
            file_names = sorted(os.listdir(self.data_directory))
            for file in file_names:
                if file.endswith('.fit') or file.endswith('.fits') or file.endswith('.fit.gz') or file.endswith('.fits.gz'):
                    self.file_paths.append(self.data_directory + file)
        else:
            if self.c1_directory is not None:
                file_names = sorted(os.listdir(self.c1_directory))
                for file in file_names:
                    if file.endswith('.fit') or file.endswith('.fits') or file.endswith('.fit.gz') or file.endswith('.fits.gz'):
                        self.file_paths.append(self.c1_directory + file)
            if self.c2_directory is not None:
                file_names = sorted(os.listdir(self.c2_directory))
                for file in file_names:
                    if file.endswith('.fit') or file.endswith('.fits') or file.endswith('.fit.gz') or file.endswith('.fits.gz'):
                        self.file_paths.append(self.c2_directory + file)
            if self.c3_directory is not None:
                file_names = sorted(os.listdir(self.c3_directory))
                for file in file_names:
                    if file.endswith('.fit') or file.endswith('.fits') or file.endswith('.fit.gz') or file.endswith('.fits.gz'):
                        self.file_paths.append(self.c3_directory + file)
        
        self.ignored_files = []  # list of files to ignore (e.g., if they are corrupted or do not comply with the FITS standard)
        
        self._scan_data_directory()  # scan data directory
        
        # define colours for circling sources in catalogs
        self.colours = list(mcolors.TABLEAU_COLORS.keys())
        self.colours.pop(self.colours.index("tab:brown"))
        self.colours.pop(self.colours.index("tab:gray"))
        self.colours.pop(self.colours.index("tab:purple"))
        self.colours.pop(self.colours.index("tab:blue"))
        
        if aperture_selector is None:  
            self.aperture_selector = default_aperture_selector
        else:
            self.aperture_selector = aperture_selector
            assert callable(self.aperture_selector), "[OPTICAM] Aperture selector must be callable."
        
        # define background calculator and write input parameters to file
        if background is None:
            self.background = Background()
            self.logger.info(f"[OPTICAM] Using default background estimator.")
        elif callable(background):
            self.background = background
            self.logger.info("[OPTICAM] Using custom background estimator.")
        else:
            raise ValueError("[OPTICAM] Background estimator must be a callable.")
        
        if local_background is None:
            self.local_background = EllipticalLocalBackground()
        elif callable(local_background):
            self.local_background = local_background
        else:
            raise ValueError("[OPTICAM] Local background estimator must be a callable.")
        
        # define source finder and write input parameters to file
        if finder == 'default':
            self.finder = Finder()
            self.logger.info(f"[OPTICAM] Using default source finder.")
        elif finder == 'crowded':
            self.finder = CrowdedFinder()
            self.logger.info(f"[OPTICAM] Using crowded source finder.")
        elif callable(finder):
            self.finder = finder
            self.logger.info("[OPTICAM] Using custom source finder.")
        else:
            raise ValueError("[OPTICAM] Source finder must be 'default', 'crowded', or a callable.")
        
        self._log_parameters()  # log input parameters
        
        self.transforms = {}  # define transforms as empty dictionary
        self.unaligned_files = []  # define unaligned files as empty list
        self.catalogs = {}  # define catalogs as empty dictionary
        
        # try to load transforms from file
        try:
            with open(self.out_directory + "cat/transforms.json", "r") as file:
                self.transforms.update(json.load(file))
            if self.verbose:
                self.logger.info("[OPTICAM] Read transforms from file.")
        except:
            pass
        
        # try to load catalogs from file
        for fltr in list(self.camera_files.keys()):
            try:
                self.catalogs.update({fltr: QTable.read(self.out_directory + f"cat/{fltr}_catalog.ecsv", format="ascii.ecsv")})
                if self.verbose:
                    print(f"[OPTICAM] Read {fltr} catalog from file.")
                continue
            except:
                pass

    def _scan_data_directory(self) -> None:
        """
        Scan the data directory for files and extract the MJD, filter, binning, and gain from each file header.
        
        Raises
        ------
        ValueError
            If more than 3 filters are found.
        ValueError
            If the binning is not consistent.
        """
        
        self.camera_files = {}  # filter : [files]
        
        # scan files in batches
        results = process_map(self._get_header_info, self.file_paths, max_workers=self.number_of_processors,
                              disable=not self.verbose, desc="[OPTICAM] Scanning data directory",
                              chunksize=len(self.file_paths) // 100)
        
        # unpack results
        filters = self._parse_header_results(results)
        
        # for each unique filter
        for fltr in np.unique(list(filters.values())):
            
            self.camera_files.update({fltr + '-band': []})  # prepare dictionary entry
            
            # for each file
            for file in self.file_paths:
                if file not in self.ignored_files:
                    # if the file filter matches the current filter
                    if filters[file] == fltr:
                        self.camera_files[fltr + '-band'].append(file)  # add file name to dict list
        
        # sort camera files so filters match camera order
        key_order = {'g-band': 0, 'u-band': 0, "g'-band": 0, "u'-band": 0, "r-band": 1, "r'-band": 1, 'i-band': 2, 'z-band': 2, "i'-band": 2, "z'-band": 2}
        self.camera_files = dict(sorted(self.camera_files.items(), key=lambda x: key_order[x[0]]))
        
        # sort files by time
        for key in list(self.camera_files.keys()):
            self.camera_files[key].sort(key=lambda x: self.mjds[x])
        
        self.t_ref = min(list(self.mjds.values()))  # get reference time
        
        # define middle image as reference image for each filter
        self.reference_indices = {}
        self.reference_files = {}
        for key in list(self.camera_files.keys()):
            self.reference_indices[key] = int(len(self.camera_files[key]) / 2)
            self.reference_files[key] = self.camera_files[key][self.reference_indices[key]]
        
        if self.verbose:
            print('[OPTICAM] Done.')
            print('[OPTICAM] Binning: ' + self.binning)
            print('[OPTICAM] Filters: ' + ', '.join(list(self.camera_files.keys())))

    def _get_header_info(self, file: str) -> Tuple[float | None, ArrayLike | None, str | None, str | None, float | None]:
        """
        Get the MJD, filter, binning, and gain from a file header.
        
        Parameters
        ----------
        file : str
            The file path.
        
        Returns
        -------
        Tuple[float, float, str, str, float]
            The MJD, BDT, filter, binning, and gain dictionaries.
        
        Raises
        ------
        KeyError
            If the file header does not contain the required keys.
        """
        
        try:
            with fits.open(file) as hdul:
                binning = hdul[0].header["BINNING"]
                gain = hdul[0].header["GAIN"]
                
                try:
                    ra = hdul[0].header["RA"]
                    dec = hdul[0].header["DEC"]
                except:
                    self.logger.info(f"[OPTICAM] Could not find RA and DEC keys in {file} header.")
                    pass
                
                mjd = get_time(hdul, file)
                
                # separate files by filter
                fltr = hdul[0].header["FILTER"]
            
            try:
                # try to compute barycentric dynamical time
                coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                bdt = apply_barycentric_correction(mjd, coords)
            except:
                bdt = mjd
                self.logger.info(f"[OPTICAM] Could not compute BDT for {file}.")
        except:
            self.logger.info(f'[OPTICAM] Skipping file {file} because it could not be read. This is usually due to the \
                file not conforming to the FITS standard, or the file being corrupted. Consider removing this file \
                    from the data directory and re-running.')
            self.ignored_files.append(file)
            return None, None, None, None, None
        
        return mjd, bdt, fltr, binning, gain

    def _parse_header_results(self, results: Tuple[float, float, str, str, float]) -> Dict[str, str]:
        """
        Parse the results returned by self._get_header_info().
        
        Parameters
        ----------
        results : Tuple
            The results.  
        
        Returns
        -------
        Tuple[str, str]
            The filter dictionary (file : filter).
        
        Raises
        ------
        ValueError
            If more than 3 filters are found.
        ValueError
            If the binning is not consistent.
        """
        
        self.mjds = {}
        self.bdts = {}
        filters = {}
        binnings = {}
        self.gains = {}
        
        # unpack results
        raw_mjds, raw_bdts, raw_filters, raw_binnings, raw_gains = zip(*results)
        
        # consolidate results
        for i in range(len(raw_mjds)):
            if self.file_paths[i] not in self.ignored_files:
                self.mjds.update({self.file_paths[i]: raw_mjds[i]})
                self.bdts.update({self.file_paths[i]: raw_bdts[i]})
                filters.update({self.file_paths[i]: raw_filters[i]})
                binnings.update({self.file_paths[i]: raw_binnings[i]})
                self.gains.update({self.file_paths[i]: raw_gains[i]})
        
        # ensure there are no more than three filters
        unique_filters = np.unique(list(filters.values()))
        if unique_filters.size > 3:
            log_filters([file_path for file_path in self.file_paths if file_path not in self.ignored_files],
                        self.out_directory)
            raise ValueError(f"[OPTICAM] More than three filters found. Image filters have been logged to {self.out_directory}diag/filters.json.")
        
        # ensure there is at most one type of binning
        unique_binning = np.unique(list(binnings.values()))
        if len(unique_binning) > 1:
            log_binnings([file_path for file_path in self.file_paths if file_path not in self.ignored_files],
                         self.out_directory)
            raise ValueError(f"[OPTICAM] Inconsistent binning detected. All images must have the same binning. Image binnings have been logged to {self.out_directory}diag/binnings.json.")
        else:
            self.binning = unique_binning[0]
            self.binning_scale = int(self.binning[0])
        
        return filters

    def _log_parameters(self):
        """
        Log any and all object parameters to a JSON file.
        """
        
        def recursive_log(param: Any, depth: int = 0, max_depth: int = 5) -> Any:
            """
            Recursively log parameters.
            
            Parameters
            ----------
            param : Any
                The parameter to log.
            depth : int, optional
                The parameter depth, by default 0.
            max_depth : int, optional
                The maximum parameter depth, by default 5. This prevents infinite recursion.
            
            Returns
            -------
            Any
                The logged parameter.
            """
            
            if depth > max_depth:
                return f"<Max depth ({max_depth}) reached>"
            
            if isinstance(param, FunctionType):
                # return function name
                return param.__name__
            if isinstance(param, (int, float, str, bool, type(None))):
                return param
            if isinstance(param, (list, tuple, set)):
                return type(param)(recursive_log(item, depth + 1, max_depth) for item in param)
            if isinstance(param, dict):
                return {key: recursive_log(value, depth + 1, max_depth) for key, value in param.items()}
            if hasattr(param, '__dict__'):
                return {key: recursive_log(value, depth + 1, max_depth) for key, value in vars(param).items()}
            return str(param)
        
        # get parameters
        params = dict(recursive_log(self, max_depth=5))
        
        params.update({'filters': list(self.camera_files.keys())})
        
        # remove some parameters that are either already saved elsewhere or are not needed
        params.pop('logger')
        params.pop('mjds')
        params.pop('bdts')
        params.pop('gains')
        params.pop('camera_files')
        params.pop('colours')
        params.pop('file_paths')
        
        # if resuming from a previous run, remove some additional parameters that are already saved elsewhere
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
        with open(self.out_directory + "misc/input_parameters.json", "w") as file:
            json.dump(params, file, indent=4)


    def get_data(self, file: str, return_error: bool = False) -> NDArray | Tuple[NDArray, NDArray]:
        """
        Get data from a file.
        
        Parameters
        ----------
        file : str
            Directory path to file.
        return_error : bool, optional
            Whether to return the error array, by default False.
        
        Returns
        -------
        NDArray | Tuple[NDArray, NDArray]
            The data array or the data and error arrays.
        """
        
        try:
            with fits.open(file) as hdul:
                data = np.array(hdul[0].data, dtype=np.float64)
                fltr = hdul[0].header["FILTER"] + '-band'
        except:
            raise ValueError(f"[OPTICAM] Could not open file {file}.")
        
        if return_error:
            error = np.sqrt(data*self.gains[file])
        
        if self.flat_corrector is not None:
            data = self.flat_corrector.correct(data, fltr)
        
        # remove cosmic rays if required
        if self.remove_cosmic_rays:
            data = cosmicray_lacosmic(data, gain_apply=False)[0]
        
        if self.rebin_factor > 1:
            data = rebin_image(data, self.rebin_factor)
            
            if return_error:
                error = rebin_image(error, self.rebin_factor)
        
        if return_error:
            return data, error
        
        return data

    def get_source_coords_from_image(self, image: NDArray, bkg: Background2D = None,
                                     away_from_edge: bool = False) -> NDArray:
        """
        Get an array of source coordinates from an image in descending order of source brightness.
        
        Parameters
        ----------
        image : NDArray
            The image from which to extract source coordinates.
        bkg : Background2D, optional
            The background of the image, by default None. If None, the background is estimated from the image. Including
            this parameter can prevent the background from being estimated multiple times.
        away_from_edge : bool, optional
            Whether to exclude sources near the edge of the image, by default False.
        
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
        tbl = clip_extended_sources(tbl)
        tbl.sort('segment_flux', reverse=True)  # sort catalog by flux in descending order
        
        coords = np.array([tbl["xcentroid"], tbl["ycentroid"]]).T
        
        if away_from_edge:
            edge = 2 * self.background.box_size
            for coord in coords:
                if coord[0] < edge or coord[0] > image.shape[1] - edge or coord[1] < edge or coord[1] > image.shape[0] - edge:
                    coords = np.delete(coords, np.where(np.all(coords == coord, axis=1)), axis=0)
        
        return np.array(coords)




    def initialise_catalogs(self, max_catalog_sources: int = 15, n_alignment_sources: int = 3,
                            transform_type: Literal['euclidean', 'similarity', 'translation'] = 'translation',
                            translation_limit: int | None = None, rotation_limit: int | None = None,
                            scaling_limit: int | None = None, overwrite: bool = False,
                            show_diagnostic_plots: bool = False) -> None:
        """
        Initialise the source catalogs for each camera. Some aspects of this method are parallelised for speed.
        
        Parameters
        ----------
        max_catalog_sources : int, optional
            The maximum number of sources included in the catalog, by default 15. Only the brightest
            max_catalog_sources sources are included in the catalog.
        n_alignment_sources : int, optional
            The number of sources to use for image alignment, by default 3. Must be >= 3. The brightest
            n_alignment_sources sources are used for image alignment.
        transform_type : Literal['euclidean', 'similarity', 'translation'], optional
            The type of transform to use for image alignment, by default 'translation'. 'translation' performs simple
            x, y translations, while 'euclidean' includes rotation and 'similarity' includes rotation and scaling.
        translation_limit : int, optional
            The maximum translation limit for image alignment, by default infinity.
        rotation_limit : int, optional
            The maximum rotation limit (in degrees) for image alignment, by default infinity.
        scaling_limit : int, optional
            The maximum scaling limit for image alignment, by default infinity.
        overwrite : bool, optional
            Whether to overwrite existing catalogs, by default False.
        show_diagnostic_plots : bool, optional
            Whether to show diagnostic plots, by default False. Diagnostic plots are saved to out_directory, so this
            parameter only affects whether the plots are displayed in the console.
        """
        
        # if catalogs already exist, skip
        if os.path.isfile(self.out_directory + 'cat/catalogs.png') and not overwrite:
            print('[OPTICAM] Catalogs already exist. To overwrite, set overwrite to True.')
            return
        
        if self.verbose:
            print('[OPTICAM] Initialising catalogs ...')
        
        if translation_limit is None:
            translation_limit = 128 // (self.binning_scale * self.rebin_factor)
        if rotation_limit is None:
            rotation_limit = 360
        if scaling_limit is None:
            scaling_limit = 1
        
        background_median = {}
        background_rms = {}
        
        stacked_images = {}
        
        # for each camera
        for fltr in self.camera_files.keys():
            
            # if no images found for camera, skip
            if len(self.camera_files[fltr]) == 0:
                continue
            
            reference_image = self.get_data(self.camera_files[fltr][self.reference_indices[fltr]])  # get reference image
            self.stacked_image = reference_image.copy()  # create stacked image
            
            try:
                reference_coords = self.get_source_coords_from_image(reference_image, away_from_edge=True)  # get source coordinates in descending order of brightness
            except:
                self.logger.info(f'[OPTICAM] No sources detected in {fltr} reference image ({self.camera_files[fltr][self.reference_indices[fltr]]}). Reducing threshold or npixels in the source finder may help.')
                continue
            
            if len(reference_coords) < n_alignment_sources:
                self.logger.info(f'[OPTICAM] Not enough sources detected in {fltr} reference image ({self.camera_files[fltr][self.reference_indices[fltr]]}) for alignment. Reducing threshold and/or n_alignment_sources may help.')
                continue
            elif len(reference_coords) > n_alignment_sources:
                reference_coords = reference_coords[:n_alignment_sources]
            
            self.logger.info(f'[OPTICAM] {fltr} alignment source coordinates: {reference_coords}')
            
            # align and stack images
            results = process_map(partial(self._align_image, reference_coords=reference_coords,
                                          n_sources=n_alignment_sources, transform_type=transform_type,
                                          translation_limit=translation_limit, rotation_limit=rotation_limit,
                                          scaling_limit=scaling_limit), self.camera_files[fltr],
                                  max_workers=self.number_of_processors, disable=not self.verbose,
                                  desc=f'[OPTICAM] Aligning {fltr} images',
                                  chunksize=len(self.camera_files[fltr]) // 100)
            self._parse_alignment_results(results, fltr)
            
            try:
                threshold = detect_threshold(self.stacked_image, nsigma=self.threshold,
                                             sigma_clip=SigmaClip(sigma=3, maxiters=10))  # estimate threshold
            except:
                self.logger.info('[OPTICAM] Unable to estimate source detection threshold for ' + fltr + ' stacked image.')
                continue
            
            try:
                # identify sources in stacked image
                segment_map = self.finder(self.stacked_image, threshold)
            except:
                self.logger.info('[OPTICAM] No sources detected in the stacked ' + fltr + ' stacked image. Reducing threshold may help.')
                continue
            
            # save stacked image and its background
            stacked_images[fltr] = self.stacked_image
            
            tbl = SourceCatalog(self.stacked_image, segment_map).to_table()  # create catalog of sources
            tbl = clip_extended_sources(tbl)  # clip extended sources
            tbl.sort('segment_flux', reverse=True)  # sort catalog by flux in descending order
            tbl = tbl[:max_catalog_sources]  # limit catalog to brightest max_catalog_sources sources
            
            # create catalog of sources in stacked image and write to file
            self.catalogs.update({fltr: tbl})
            self.catalogs[fltr].write(self.out_directory + f"cat/{fltr}_catalog.ecsv", format="ascii.ecsv",
                                            overwrite=True)
        
        del self.stacked_image  # stacked image no longer needed
        
        # compile catalog
        self._plot_catalog(stacked_images)
        
        # # diagnostic plots
        self._plot_time_between_files(show_diagnostic_plots)  # plot time between observations
        # self._plot_backgrounds(background_median, background_rms, show_diagnostic_plots)  # plot background medians and RMSs
        self._plot_background_meshes(stacked_images, show_diagnostic_plots)  # plot background meshes
        # for (fltr, stacked_image) in stacked_images.items():
        #     self._visualise_psfs(stacked_image, fltr, show_diagnostic_plots)
        
        # save transforms to file
        with open(self.out_directory + "cat/transforms.json", "w") as file:
            json.dump(self.transforms, file, indent=4)
        
        # write unaligned files to file
        if len(self.unaligned_files) > 0:
            with open(self.out_directory + "diag/unaligned_files.txt", "w") as unaligned_file:
                for file in self.unaligned_files:
                    unaligned_file.write(file + "\n")

    def _align_image(self, file: str, reference_coords: NDArray, n_sources: int,
                                     transform_type: Literal['euclidean', 'similarity', 'translation'],
                                     translation_limit: int, rotation_limit: int,
                                     scaling_limit: int) -> Tuple[List[float], float, float]:
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
        n_sources : int
            The number of sources to use for image alignment.
        transform_type : Literal['euclidean', 'similarity', 'translation']
            The type of transform to use for image alignment.
        translation_limit : int
            The maximum translation limit for image alignment.
        rotation_limit : int
            The maximum rotation limit (in degrees) for image alignment.
        scaling_limit : int
            The maximum scaling limit for image alignment.
        
        Returns
        -------
        Tuple[List[float], float, float]
            The transform parameters, background median, and background RMS.
        """
        
        data = self.get_data(file)  # get image data
        
        bkg = self.background(data)  # get background
        background_median = bkg.background_median
        background_rms = bkg.background_rms_median
        
        data_clean = data - bkg.background  # remove background from image
        
        try:
            coords = self.get_source_coords_from_image(data_clean)  # get source coordinates in descending order of brightness
        except:
            self.logger.info('[OPTICAM] No sources detected in ' + file + '. Reducing threshold or npixels in the source finder may help.')
            return None, None, None, file
        
        distance_matrix = cdist(reference_coords, coords)  # compute distance matrix
        try:
            reference_indices, indices = linear_sum_assignment(distance_matrix)  # solve assignment problem
        except:
            self.logger.info('[OPTICAM] Could not align ' + file + '. Reducing threshold and/or n_alignment_sources may help.')
            return None, None, None, file
        
        # compute transform
        if transform_type == 'translation':
            dx = np.mean(coords[indices, 0] - reference_coords[reference_indices, 0])
            dy = np.mean(coords[indices, 1] - reference_coords[reference_indices, 1])
            if abs(dx) < translation_limit and abs(dy) < translation_limit:
                transform = SimilarityTransform(translation=[dx, dy])
            else:
                self.logger.info(f'[OPTICAM] File {file} exceeded translation limit. Translation limit is {translation_limit:.1f}, but translation was ({dx:.1f}, {dy:.1f}).')
                return None, None, None, file
        else:
            transform = estimate_transform(transform_type, reference_coords[reference_indices], coords[indices])
        
        # transform and stack image
        self.stacked_image += warp(data_clean, transform.inverse, output_shape=self.stacked_image.shape, order=3,
                                   mode='constant', cval=np.nanmedian(data), clip=True, preserve_range=True)
        
        return transform.params.tolist(), background_median, background_rms
    
    def _parse_alignment_results(self, results, fltr: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Parse the results of image alignment.
        
        Parameters
        ----------
        results :
            The results.
        fltr : str
            The filter.
        
        Returns
        -------
        Tuple[Dict[str, float], Dict[str, float]]
            The background medians, and background RMSs.
        """
        
        transforms = {}
        unaligned_files = []
        background_medians = {}
        background_rmss = {}
        
        # unpack results
        raw_transforms, raw_background_medians, raw_background_rmss = zip(*results)
        
        for i in range(len(self.camera_files[fltr])):
            if raw_transforms[i] is None:
                unaligned_files.append(self.camera_files[fltr][i])
            else:
                transforms.update({self.camera_files[fltr][i]: raw_transforms[i]})
                background_medians.update({self.camera_files[fltr][i]: raw_background_medians[i]})
                background_rmss.update({self.camera_files[fltr][i]: raw_background_rmss[i]})
        
        self.transforms.update(transforms)  # update transforms
        self.unaligned_files += unaligned_files  # update unaligned files
        
        if self.verbose:
            print(f"[OPTICAM] Done. {len(unaligned_files)} image(s) could not be aligned.")
        
        return background_medians, background_rmss
    
    def _plot_catalog(self, stacked_images: Dict[str, NDArray]) -> None:
        """
        Plot the source catalogs on top of the stacked images
        
        Parameters
        ----------
        stacked_images : Dict[str, NDArray]
            The stacked images for each camera.
        """
        
        fig, ax = plt.subplots(ncols=len(self.catalogs), tight_layout=True, figsize=(len(stacked_images) * 5, 5))
        
        if len(self.catalogs) == 1:
            ax = [ax]
        
        for i, fltr in enumerate(list(self.catalogs.keys())):
            
            plot_image = np.clip(stacked_images[fltr], 0, None)  # clip negative values to zero for better visualisation
            
            # plot stacked image
            ax[i].imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                            norm=simple_norm(plot_image, stretch="log"))
            
            # get aperture radius
            radius = self.scale*self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
                     
            for j in range(len(self.catalogs[fltr])):
                # label sources
                ax[i].add_patch(Circle(xy=(self.catalogs[fltr]["xcentroid"][j],
                                            self.catalogs[fltr]["ycentroid"][j]),
                                        radius=radius, edgecolor=self.colours[j % len(self.colours)], 
                                        facecolor="none", lw=1))
                ax[i].add_patch(Circle(xy=(self.catalogs[fltr]["xcentroid"][j],
                                            self.catalogs[fltr]["ycentroid"][j]),
                                       radius=self.local_background.r_in_scale*radius,
                                       edgecolor=self.colours[j % len(self.colours)], facecolor="none", lw=1, ls=":"))
                ax[i].add_patch(Circle(xy=(self.catalogs[fltr]["xcentroid"][j], 
                                            self.catalogs[fltr]["ycentroid"][j]),
                                        radius=self.local_background.r_out_scale*radius,
                                        edgecolor=self.colours[j % len(self.colours)], facecolor="none", lw=1,
                                        ls=":"))
                ax[i].text(self.catalogs[fltr]["xcentroid"][j] + 1.05*radius,
                            self.catalogs[fltr]["ycentroid"][j] + 1.05*radius, j + 1, 
                            color=self.colours[j % len(self.colours)])
                
                # label plot
                ax[i].set_title(fltr)
                ax[i].set_xlabel("X")
                ax[i].set_ylabel("Y")
        
        fig.savefig(self.out_directory + "cat/catalogs.png")
        
        if self.show_plots:
            plt.show(fig)
        else:
            fig.clear()
            plt.close(fig)

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
        
        fig.savefig(self.out_directory + "diag/background_meshes.png")
        
        if show and self.show_plots:
            plt.show(fig)
        else:
            fig.clear()
            plt.close(fig)

    def _plot_time_between_files(self, show: bool) -> None:
        """
        Plot the times between each file for each camera.
        
        Parameters
        ----------
        show : bool
            Whether to display the plot.
        """
        
        fig, axs = plt.subplots(nrows=2, ncols=len(self.catalogs), tight_layout=True, figsize=((2 * len(self.catalogs) / 3) * 6.4, 2 * 4.8))
        
        for fltr in list(self.catalogs.keys()):
            times = np.array([self.mjds[file] for file in self.camera_files[fltr]])
            times -= times.min()
            times *= 86400  # convert to seconds from first observation
            dt = np.diff(times)  # get time between files
            file_numbers = np.arange(2, len(times) + 1, 1)  # start from 2 because we are plotting the time between files
            
            bin_edges = np.arange(0, int(dt.max()) + 2, 1)  # define bins with width 1 s
            
            if len(self.catalogs) == 1:
                axs[0].set_title(fltr)
                axs[0].plot(file_numbers, dt, "k-", lw=1)
                
                axs[1].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
                axs[1].set_yscale("log")
                
                axs[0].set_ylabel("Time between files [s]")
                axs[0].set_xlabel("File number")
                axs[1].set_ylabel("N")
                axs[1].set_xlabel("Time between files [s]")
            else:
                axs[0, list(self.catalogs.keys()).index(fltr)].set_title(fltr)
                axs[0, list(self.catalogs.keys()).index(fltr)].plot(file_numbers, dt, "k-", lw=1)
                
                axs[1, list(self.catalogs.keys()).index(fltr)].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
                axs[1, list(self.catalogs.keys()).index(fltr)].set_yscale("log")
                
                axs[0, 0].set_ylabel("Time between files [s]")
                
                for col in range(len(self.catalogs)):
                    axs[0, col].set_xlabel("File number")
                    
                    axs[1, col].set_xlabel("Time between files [s]")
        
        for ax in axs.flatten():
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
        fig.savefig(self.out_directory + "diag/header_times.png")
        
        if show and self.show_plots:
            plt.show(fig)
        else:
            fig.clear()
            plt.close(fig)

    def _plot_backgrounds(self, background_median: Dict[str, List], background_rms: Dict[str, List], show: bool) -> None:
        """
        Plot the time-varying background for each camera.
        
        Parameters
        ----------
        background_median : Dict[str, List]
            The median background for each camera.
        background_rms : Dict[str, List]
            The background RMS for each camera.
        show: bool
            Whether to display the plot.
        """
        
        fig, axs = plt.subplots(nrows=2, ncols=len(self.catalogs), tight_layout=True, figsize=((2 * len(self.catalogs) / 3) * 6.4, 2 * 4.8), sharex='col')
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            mjds = np.array([self.mjds[file] for file in self.camera_files[fltr]])
            bdts = np.array([self.bdts[file] for file in self.camera_files[fltr]])
            plot_times = (mjds - self.t_ref)*86400  # convert to seconds from first observation
            
            if len(self.catalogs) == 1:
                axs[0].set_title(fltr)
                axs[0].plot(plot_times, background_rms[fltr], "k.", ms=2)
                axs[1].plot(plot_times, background_median[fltr], "k.", ms=2)
                
                axs[1].set_xlabel(f"Time from MJD {mjds.min():.4f} [s]")
                axs[0].set_ylabel("Median background RMS")
                axs[1].set_ylabel("Median background")
            else:
                # plot background
                axs[0, list(self.catalogs.keys()).index(fltr)].set_title(fltr)
                axs[0, list(self.catalogs.keys()).index(fltr)].plot(plot_times, background_rms[fltr], "k.", ms=2)
                axs[1, list(self.catalogs.keys()).index(fltr)].plot(plot_times, background_median[fltr], "k.", ms=2)
                
                for col in range(len(self.catalogs)):
                    axs[1, col].set_xlabel(f"Time from MJD {mjds.min():.4f} [s]")
                
                axs[0, 0].set_ylabel("Median background RMS")
                axs[1, 0].set_ylabel("Median background")
            
            # write background to file
            with open(self.out_directory + 'diag/' + fltr + '_background.csv', 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['MJD', 'BDT', 'RMS', 'median'])
                for i in range(len(self.camera_files[fltr])):
                    writer.writerow([mjds[i], bdts[i], background_rms[fltr][i], background_median[fltr][i]])
        
        for ax in axs.flatten():
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
        # save plot
        fig.savefig(self.out_directory + "diag/background.png")
        
        # either show or close plot
        if show and self.show_plots:
            plt.show()
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
            
            fig.savefig(self.out_directory + f'diag/{fltr}_source_{source}_psf.png')
            
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
            elif os.path.exists(self.out_directory + f"cat/{fltr}_images.gif") and not overwrite:
                print(f"[OPTICAM] {fltr} GIF already exists. To overwrite, set overwrite to True.")
                continue
            
            # create gif frames directory if it does not exist
            if not os.path.isdir(self.out_directory + f"diag/{fltr}_gif_frames"):
                os.mkdir(self.out_directory + f"diag/{fltr}_gif_frames")
            
            process_map(partial(self._create_gif_frames, fltr=fltr), self.camera_files[fltr],
                        max_workers=self.number_of_processors, disable=not self.verbose,
                        desc=f"[OPTICAM] Creating {fltr} GIF frames", chunksize=len(self.camera_files[fltr]) // 100)
            
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
        
        data = self.get_data(file)
        
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
            
            radius = self.scale*self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            
            ax.add_patch(Circle(xy=(aperture_position), radius=radius,
                                    edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1))
            ax.add_patch(Circle(xy=(aperture_position), radius=self.local_background.r_in_scale*radius,
                                edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1, ls=":"))
            ax.add_patch(Circle(xy=(aperture_position), radius=self.local_background.r_out_scale*radius,
                                edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1, ls=":"))
            ax.text(aperture_position[0] + 1.05*radius, aperture_position[1] + 1.05*radius, i + 1,
                        color=self.colours[i % len(self.colours)])
        
        ax.set_title(title, color=colour)
        
        fig.savefig(self.out_directory + 'diag/' + fltr + '_gif_frames/' + file_name + '.png')

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
                frames.append(Image.open(self.out_directory + 'diag/' + fltr + '_gif_frames/' + file.split('/')[-1].split(".")[0] + '.png'))
            except:
                pass
        
        # save gif
        frames[0].save(self.out_directory + 'cat/' + fltr + '_images.gif', format='GIF', append_images=frames[1:], 
                       save_all=True, duration=200, loop=0)
        
        # close images
        for frame in frames:
            frame.close()
        del frames
        
        # delete frames after gif is saved
        if not keep_frames:
            for file in tqdm(os.listdir(self.out_directory + f"diag/{fltr}_gif_frames"), disable=not self.verbose,
                             desc=f"[OPTICAM] Deleting {fltr} GIF frames"):
                os.remove(self.out_directory + f"diag/{fltr}_gif_frames/{file}")




    def forced_photometry(self, phot_type: Literal["aperture", "annulus", "both"] = "both",
                          overwrite: bool = False) -> None:
        """
        Perform forced photometry on the images in out_directory. The light curves produced by this method are generally
        going to have lower signal-to-noise ratios than those produced by the photometry() method, but they have the
        benefit of being able to extract light curves for sources that are not detected in all images.
        
        Parameters
        ----------
        phot_type : Literal["aperture", "annulus", "both"], optional
            The type of photometry to perform, by default "both". If "aperture", only aperture photometry is performed.
            If "annulus", only annulus photometry is performed. If "both", both aperture and annulus photometry are
            performed simultaneously (this is more efficient that performing both separately since it only opens the
            file once).
        overwrite : bool, optional
            Whether to overwrite existing light curves, by default False.
        
        Raises
        ------
        ValueError
            If phot_type is not recognised.
        """
        
        assert phot_type in ['aperture', 'annulus', 'both'], f"[OPTICAM] Photometry type {phot_type} not recognised."
        
        # create output directories if they do not exist
        if phot_type == 'aperture':
            if not os.path.isdir(self.out_directory + f"aperture_light_curves"):
                os.mkdir(self.out_directory + f"aperture_light_curves")
        elif phot_type == 'annulus':
            if not os.path.isdir(self.out_directory + f"annulus_light_curves"):
                os.mkdir(self.out_directory + f"annulus_light_curves")
        else:
            if not os.path.isdir(self.out_directory + f"aperture_light_curves"):
                os.mkdir(self.out_directory + f"aperture_light_curves")
            if not os.path.isdir(self.out_directory + f"annulus_light_curves"):
                os.mkdir(self.out_directory + f"annulus_light_curves")
        
        self.logger.info(f'[OPTICAM] Performing forced photometry with phot_type={phot_type} ...')
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            # get list of possible light curve files
            if phot_type == 'aperture':
                light_curve_files = [self.out_directory + f"aperture_light_curves/{fltr}_source_{i}.csv" for i in range(len(self.catalogs[fltr]))]
            elif phot_type == 'annulus':
                light_curve_files = [self.out_directory + f"annulus_light_curves/{fltr}_source_{i}.csv" for i in range(len(self.catalogs[fltr]))]
            else:
                light_curve_files = [self.out_directory + f"aperture_light_curves/{fltr}_source_{i}.csv" for i in range(len(self.catalogs[fltr]))]
                light_curve_files += [self.out_directory + f"annulus_light_curves/{fltr}_source_{i}.csv" for i in range(len(self.catalogs[fltr]))]
            
            # check if light curves already exist
            if all([os.path.isfile(file) for file in light_curve_files]) and not overwrite:
                self.logger.info(f'[OPTICAM] {fltr} light curves already exist and overwrite is False. Skipping ...')
                continue
            
            # get aperture radius
            try:
                radius = self.scale*self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            except:
                # skip cameras with no sources
                continue
            
            self.logger.info(f'[OPTICAM] {fltr} forced photometry aperture radius: {radius} pixels.')
            
            results = process_map(partial(self._perform_forced_photometry, fltr=fltr, radius=radius,
                                          phot_type=phot_type), self.camera_files[fltr],
                                  max_workers=self.number_of_processors,
                                  desc=f"[OPTICAM] Performing forced photometry on {fltr} images",
                                  disable=not self.verbose, chunksize=len(self.camera_files[fltr]) // 100)
            
            self._save_forced_photometry_results(results, phot_type, fltr)

    def _perform_forced_photometry(self, file: str, fltr: str, radius: float,
                                   phot_type: Literal["aperture", "annulus", "both"]):
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
        phot_type : Literal[&quot;aperture&quot;, &quot;annulus&quot;, &quot;both&quot;]
            The type of photometry to performn.
        
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
        if file == self.camera_files[fltr][self.reference_indices[fltr]]:
            flag = 'A'
        elif file not in self.transforms.keys():
            flag = 'B'
        else:
            flag = 'A'
            transform = self.transforms[file]
        
        # get image data and its error
        data, error = self.get_data(file, return_error=True)
        
        # get background subtracted image and its error if required
        if phot_type in ['aperture', 'both']:
            bkg = self.background(data)
            clean_data = data - bkg.background
            clean_error = calc_total_error(clean_data, bkg.background_rms, error)
        
        # for each source
        for i in range(len(self.catalogs[fltr])):
            # load the source's catalog position
            catalog_position = (self.catalogs[fltr]["xcentroid"][i], self.catalogs[fltr]["ycentroid"][i])
            
            # try to transform the catalog position using the image transform, otherwise use the catalog position
            try:
                position = matrix_transform(catalog_position, transform)[0]
            except:
                position = catalog_position
            
            # perform photometry
            if phot_type == 'aperture':
                flux, flux_error = self._compute_aperture_flux(clean_data, clean_error, position, radius)
                aperture_fluxes.append(flux)
                aperture_flux_errors.append(flux_error)
            elif phot_type == 'annulus':
                flux, flux_error, local_background, local_background_error, local_background_per_pixel, local_background_error_per_pixel = self._compute_annulus_flux(data, error, position, radius)
                annulus_fluxes.append(flux)
                annulus_flux_errors.append(flux_error)
                local_backgrounds.append(local_background)
                local_background_errors.append(local_background_error)
                local_backgrounds_per_pixel.append(local_background_per_pixel)
                local_background_errors_per_pixel.append(local_background_error_per_pixel)
            else:
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
        
        # return the results in the correct format for the specified phot_type
        if phot_type == 'aperture':
            return aperture_fluxes, aperture_flux_errors, flag
        elif phot_type == 'annulus':
            return annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flag
        else:
            return aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flag

    def _parse_forced_photometry_results(self, results, phot_type: Literal["aperture", "annulus", "both"]):
        """
        Parse the forced photometry photometric results.
        
        Parameters
        ----------
        results :
            The photometric results.
        phot_type : Literal['aperture', 'annulus', 'both']
            The type of photometry that has been performed.
        
        Returns
        -------
        Tuple
            The parsed photometric results.
        """
        
        if phot_type == 'aperture':
            aperture_fluxes, aperture_flux_errors, flags = zip(*results)
            
            return list(aperture_fluxes), list(aperture_flux_errors), list(flags)
        
        elif phot_type == 'annulus':
            annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = zip(*results)
            
            return list(annulus_fluxes), list(annulus_flux_errors), list(local_backgrounds), list(local_background_errors), list(local_backgrounds_per_pixel), list(local_background_errors_per_pixel), list(flags)
        
        else:
            aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = zip(*results)
            
            return list(aperture_fluxes), list(aperture_flux_errors), list(annulus_fluxes), list(annulus_flux_errors), list(local_backgrounds), list(local_background_errors), list(local_backgrounds_per_pixel), list(local_background_errors_per_pixel), list(flags)

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
        local_background_per_pixel, local_background_error_per_pixel = self.local_background(data, error, radius, radius, 0, position)
        
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

    def _save_forced_photometry_results(self, results: Tuple, phot_type: str, fltr: str) -> None:
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
        mjds = [self.mjds[file] for file in self.camera_files[fltr]]
        bdts = [self.bdts[file] for file in self.camera_files[fltr]]
        
        # save light curves
        if phot_type == 'aperture':
            # parse results
            aperture_fluxes, aperture_flux_errors, flags = self._parse_forced_photometry_results(results, phot_type)
            
            # save light curves
            for i in tqdm(range(len(self.catalogs[fltr])), disable=not self.verbose, desc=f"[OPTICAM] Saving {fltr} light curves"):
                self._save_aperture_light_curve(mjds, bdts, aperture_fluxes, aperture_flux_errors, flags, fltr, i)
        elif phot_type == 'annulus':
            # parse results
            annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = self._parse_forced_photometry_results(results, phot_type)
            
            # save light curves
            for i in tqdm(range(len(self.catalogs[fltr])), disable=not self.verbose, desc=f"[OPTICAM] Saving {fltr} light curves"):
                self._save_annulus_light_curve(mjds, bdts, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags, fltr, i)
        else:
            # parse results
            aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = self._parse_forced_photometry_results(results, phot_type)
            
            # save light curves
            for i in tqdm(range(len(self.catalogs[fltr])), disable=not self.verbose, desc=f"[OPTICAM] Saving {fltr} light curves"):
                self._save_aperture_light_curve(mjds, bdts, aperture_fluxes, aperture_flux_errors, flags, fltr, i)
                self._save_annulus_light_curve(mjds, bdts, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags, fltr, i)

    def _save_aperture_light_curve(self, mjds: List[float], bdts: List[float], fluxes: List[List[float]],
                                   flux_errors: List[List[float]], flags: List[str], fltr: str,
                                   source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        mjds : List[float]
            The observation MJDs.
        bdts : List[float]
            The observation BDTs.
        fluxes : List[List[float]]
            The source fluxes.
        flux_errors : List[List[float]]
            The source flux errors.
        flags : List[str]
            The quality flags.
        fltr : str
            The filter.
        source_index : int
            The source index, not to be confused with the source number. The source index is one less than the source
            number.
        """
        
        with open(self.out_directory + f"aperture_light_curves/{fltr}_source_{source_index + 1}.csv", "w") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(["MJD", 'BDT', "flux", "flux_error", "quality_flag"])
            
            # for each observation
            for i in range(len(self.camera_files[fltr])):
                csvwriter.writerow([mjds[i], bdts[i], fluxes[i][source_index], flux_errors[i][source_index], flags[i]])
        
        # load light curve from file
        df = pd.read_csv(self.out_directory + f"aperture_light_curves/{fltr}_source_{source_index + 1}.csv")
        aligned_mask = df["quality_flag"] == "A"  # mask for aligned observations
        
        # reformat MJD to seconds from first observation
        df["time"] = df["MJD"] - self.t_ref
        df["time"] *= 86400
        
        fig, ax = plt.subplots(num=1, clear=True, tight_layout=True, figsize=(6.4, 4.8))
        
        ax.errorbar(df["time"].values[aligned_mask], df["flux"].values[aligned_mask], yerr=df["flux_error"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
        ax.errorbar(df["time"].values[~aligned_mask], df["flux"].values[~aligned_mask], yerr=df["flux_error"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
        ax.set_ylabel("Flux [counts]")
        ax.set_title(f"{fltr} Source {source_index + 1}")
        ax.set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
        
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        # save light curve plot to file
        fig.savefig(self.out_directory + f"aperture_light_curves/{fltr}_source_{source_index + 1}.png")
        fig.clear()
        plt.close(fig)

    def _save_annulus_light_curve(self, mjds: List[float], bdts: List[float], fluxes: List[List[float]],
                                  flux_errors: List[List[float]], local_backgrounds: List[List[float]],
                                  local_background_errors: List[List[float]],
                                  local_backgrounds_per_pixel: List[List[float]],
                                  local_background_errors_per_pixel: List[List[float]],
                                  flags: List[str], fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        mjds : List[List[float]]
            The observation MJDs.
        bdts : List[List[float]]
            The observation BDTs.
        fluxes : List[List[float]]
            The source fluxes.
        flux_errors : List[List[float]]
            The source flux errors.
        local_backgrounds : List[List[float]]
            The local backgrounds.
        local_background_errors : List[List[float]]
            The local background errors.
        local_backgrounds_per_pixel : List[List[float]]
            The local backgrounds per pixel.
        local_background_errors_per_pixel : List[List[float]]
            The local background errors per pixel.
        flags : List[str]
            The quality flags.
        fltr : str
            The filter.
        source_index : int
            The source index, not to be confused with the source number. The source index is one less than the source
            number.
        """
        
        # save source light curve to file
        with open(self.out_directory + f"annulus_light_curves/{fltr}_source_{source_index + 1}.csv", "w") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(['MJD', 'BDT', "flux", "flux_error", "local_background", "local_background_error",
                                "local_background_per_pixel", "local_background_error_per_pixel", "quality_flag"])
            for i in range(len(self.camera_files[fltr])):
                csvwriter.writerow([mjds[i], bdts[i], fluxes[i][source_index], flux_errors[i][source_index],
                                    local_backgrounds[i][source_index], local_background_errors[i][source_index],
                                    local_backgrounds_per_pixel[i][source_index],
                                    local_background_errors_per_pixel[i][source_index], flags[i]])
        
        # load light curve from file
        df = pd.read_csv(self.out_directory + f"annulus_light_curves/{fltr}_source_{source_index + 1}.csv")
        aligned_mask = df["quality_flag"] == "A"  # mask for aligned observations
        
        # reformat MJD to seconds from first observation
        df["time"] = df["MJD"] - self.t_ref
        df["time"] *= 86400
        
        fig, axs = plt.subplots(num=1, clear=True, nrows=3, tight_layout=True, figsize=(6.4, 2*4.8), sharex=True,
                                gridspec_kw={"hspace": 0})
        
        axs[0].errorbar(df["time"].values[aligned_mask], df["flux"].values[aligned_mask], yerr=df["flux_error"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
        axs[0].errorbar(df["time"].values[~aligned_mask], df["flux"].values[~aligned_mask], yerr=df["flux_error"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
        axs[0].set_ylabel("Flux [counts]")
        axs[0].set_title(f"{fltr} Source {source_index + 1}")
        
        axs[1].errorbar(df["time"].values[aligned_mask], df["local_background_per_pixel"].values[aligned_mask], yerr=df["local_background_error_per_pixel"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
        axs[1].errorbar(df["time"].values[~aligned_mask], df["local_background_per_pixel"].values[~aligned_mask], yerr=df["local_background_error_per_pixel"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
        axs[1].set_ylabel("Local background [cts/pixel]")
        
        axs[2].plot(df["time"].values[aligned_mask], df["flux"].values[aligned_mask]/df["local_background"].values[aligned_mask], "k.", ms=2)
        axs[2].plot(df["time"].values[~aligned_mask], df["flux"].values[~aligned_mask]/df["local_background"].values[~aligned_mask], "r.", ms=2, alpha=.2)
        axs[2].set_ylabel("SNR")
        axs[2].set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
        
        for ax in axs:
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
        fig.savefig(self.out_directory + f"annulus_light_curves/{fltr}_source_{source_index + 1}.png")
        fig.clear()
        plt.close(fig)




    def photometry(self, phot_type: Literal['both', 'normal', 'optimal'] = 'both',
                   background_method: Literal['global', 'local'] = 'global', tolerance: float = 5.,
                   overwrite: bool = False) -> None:
        """
        Perform photometry by fitting for the source positions in each image. In general, this method should produce
        light curves with better signal-to-noise ratios than forced photometry. However, this method can misidentify
        sources if the field is crowded or the alignments are poor.
        
        Parameters
        ----------
        phot_type : Literal['both', 'normal', 'optimal']
            The type of photometry to perform. 'normal' will extract fluxes using simple aperture photometry, while
            'optimal' will extract fluxes using the optimal photometry method outlined in Naylor 1998, MNRAS, 296, 339.
            'both' will extract fluxes using both methods (this is more efficient than performing both separately since
            it only opens the file once).
        background_method : Literal['global', 'local'], optional
            The method to use for background subtraction, by default 'global'. 'global' uses the background attribute to
            compute the 2D background across an entire image, while 'local' uses the local_background attribute to
            estimate the local background around each source.
        tolerance : float, optional
            The tolerance for source position matching in standard deviations (assuming a Gaussian PSF), by default 5.
            This parameter defines how far from the transformed catalog position a source can be while still being
            considered the same source. If the alignments are good and/or the field is crowded, consider reducing this
            value. For poor alignments and/or uncrowded fields, this value can be increased.
        overwrite : bool, optional
            Whether to overwrite existing light curves, by default False.
        """
        
        assert phot_type in ['normal', 'optimal', 'both'], f"[OPTICAM] Photometry type {phot_type} not recognised."
        
        # create output directories if they do not exist
        if phot_type == 'normal':
            if not os.path.isdir(self.out_directory + f"normal_light_curves"):
                os.mkdir(self.out_directory + f"normal_light_curves")
        elif phot_type == 'optimal':
            if not os.path.isdir(self.out_directory + f"optimal_light_curves"):
                os.mkdir(self.out_directory + f"optimal_light_curves")
        else:
            if not os.path.isdir(self.out_directory + f"normal_light_curves"):
                os.mkdir(self.out_directory + f"normal_light_curves")
            if not os.path.isdir(self.out_directory + f"optimal_light_curves"):
                os.mkdir(self.out_directory + f"optimal_light_curves")
        
        self.logger.info(f'[OPTICAM] Performing photometry with phot_type={phot_type} ...')
        
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            # get list of possible light curve files
            if phot_type == 'normal':
                light_curve_files = [self.out_directory + f"normal_light_curves/{fltr}_source_{i + 1}.csv" for i in range(len(self.catalogs[fltr]))]
            elif phot_type == 'optimal':
                light_curve_files = [self.out_directory + f"optimal_light_curves/{fltr}_source_{i + 1}.csv" for i in range(len(self.catalogs[fltr]))]
            else:
                light_curve_files = [self.out_directory + f"normal_light_curves/{fltr}_source_{i + 1}.csv" for i in range(len(self.catalogs[fltr]))]
                light_curve_files += [self.out_directory + f"optimal_light_curves/{fltr}_source_{i + 1}.csv" for i in range(len(self.catalogs[fltr]))]
            
            # check if light curves already exist
            if all([os.path.isfile(file) for file in light_curve_files]) and not overwrite:
                self.logger.info(f'[OPTICAM] {fltr} light curves already exist and overwrite is False. Skipping ...')
                continue
            
            # get PSF parameters
            semimajor_sigma = self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            semiminor_sigma = self.aperture_selector(self.catalogs[fltr]["semiminor_sigma"].value)
            
            self.logger.info(f'[OPTICAM] {fltr} semi-major axis of PSF: {semimajor_sigma} pixels.')
            self.logger.info(f'[OPTICAM] {fltr} semi-minor axis of PSF: {semiminor_sigma} pixels.')
            self.logger.info(f'[OPTICAM] {fltr} normal aperture semi-major axis: {self.fwhm_scale * semimajor_sigma} pixels.')
            self.logger.info(f'[OPTICAM] {fltr} normal aperture semi-minor axis: {self.fwhm_scale * semiminor_sigma} pixels.')
            
            results = process_map(partial(self._perform_photometry_on_batch, fltr=fltr, semimajor_sigma=semimajor_sigma,
                                          semiminor_sigma=semiminor_sigma, background_method=background_method,
                                          tolerance=tolerance, phot_type=phot_type), self.camera_files[fltr],
                                  max_workers=self.number_of_processors,
                                  desc=f"[OPTICAM] Performing photometry on {fltr} images",
                                  disable=not self.verbose, chunksize=len(self.camera_files[fltr]) // 100)
            
            self._save_photometry_results(results, phot_type, fltr)

    def _perform_photometry_on_batch(self, file: str, fltr: str, semimajor_sigma: float, semiminor_sigma: float,
                                    background_method: Literal['global', 'local'], tolerance: float,
                                    phot_type: Literal['normal', 'optimal', 'both']):
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
        if file not in self.transforms.keys() and file != self.camera_files[fltr][self.reference_indices[fltr]]:
            if phot_type in ['normal', 'optimal']:
                return None, None, None, None, None
            else:
                return None, None, None, None, None, None, np.zeros(len(self.catalogs[fltr]))
        
        # define lists to store results for each file
        normal_fluxes, normal_flux_errors = [], []
        optimal_fluxes, optimal_flux_errors = [], []
        detections = np.zeros(len(self.catalogs[fltr]))
        
        # get image data
        data, error = self.get_data(file, return_error=True)
        
        # get background subtracted image for source detection
        bkg = self.background(data)
        clean_data = data - bkg.background
        
        if background_method == 'global':
            # combine error in background subtracted image
            error = np.sqrt(error**2 + bkg.background_rms**2)
        
        # find sources in the background subtracted image
        try:
            segment_map = self.finder(clean_data, threshold=self.threshold * bkg.background_rms)
        except:
            # if no sources are found, return None for all results
            if phot_type in ['normal', 'optimal']:
                return None, None, None, None, None
            else:
                return None, None, None, None, None, None, np.zeros(len(self.catalogs[fltr]))
        
        # create source table
        file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
        file_tbl = file_cat.to_table()
        
        # for each source in the catalog
        for i in range(len(self.catalogs[fltr])):
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
            if phot_type == 'normal':
                if background_method == 'global':
                    # compute normal flux using global background
                    flux, flux_error = self._compute_normal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]['orientation'][i].value)
                else:
                    # compute normal flux using local background
                    flux, flux_error = self._compute_normal_flux(data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]['orientation'][i].value, True)
                
                # append results to lists
                normal_fluxes.append(flux)
                normal_flux_errors.append(flux_error)
            
            elif phot_type == 'optimal':
                if background_method == 'global':
                    # compute optimal flux using global background
                    flux, flux_error = self._compute_optimal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]['orientation'][i].value)
                else:
                    # compute optimal flux using local background
                    flux, flux_error = self._compute_optimal_flux(data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]['orientation'][i].value, True)
                
                # append results to lists
                optimal_fluxes.append(flux)
                optimal_flux_errors.append(flux_error)
            
            else:
                if background_method == 'global':
                    # compute normal flux using global background
                    flux, flux_error = self._compute_normal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]['orientation'][i].value)
                else:
                    # compute normal flux using local background
                    flux, flux_error = self._compute_normal_flux(data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]['orientation'][i].value, True)
                
                # append results to lists
                normal_fluxes.append(flux)
                normal_flux_errors.append(flux_error)
                
                if background_method == 'global':
                    # compute optimal flux using global background
                    flux, flux_error = self._compute_optimal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]['orientation'][i].value)
                else:
                    # compute optimal flux using local background
                    flux, flux_error = self._compute_optimal_flux(data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]['orientation'][i].value, True)
                
                # append results to lists
                optimal_fluxes.append(flux)
                optimal_flux_errors.append(flux_error)
        
        # return the results in the correct format for the specified phot_type
        if phot_type == 'normal':
            return self.mjds[file], self.bdts[file], normal_fluxes, normal_flux_errors, detections
        elif phot_type == 'optimal':
            return self.mjds[file], self.bdts[file], optimal_fluxes, optimal_flux_errors, detections
        else:
            return self.mjds[file], self.bdts[file], normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, detections

    def _parse_photometry_results(self, results, phot_type: Literal['normal', 'optimal', 'both']):
        """
        Parse the photometry results.
        
        Parameters
        ----------
        results :
            The photometry results.
        phot_type : Literal[&#39;normal&#39;, &#39;optimal&#39;, &#39;both&#39;]
            The type of photometry that has been performed.
        
        Returns
        -------
        Tuple
            The parsed photometric results.
        """
        
        if phot_type == 'normal':
            mjds, bdts, normal_fluxes, normal_flux_errors, detections = zip(*results)
            return list(mjds), list(bdts), list(normal_fluxes), list(normal_flux_errors), np.sum(detections, axis=0)
        elif phot_type == 'optimal':
            mjds, bdts, optimal_fluxes, optimal_flux_errors, detections = zip(*results)
            return list(mjds), list(bdts), list(optimal_fluxes), list(optimal_flux_errors), np.sum(detections, axis=0)
        else:
            mjds, bdts, normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, detections = zip(*results)
            return list(mjds), list(bdts), list(normal_fluxes), list(normal_flux_errors), list(optimal_fluxes), list(optimal_flux_errors), np.sum(detections, axis=0)

    def _save_photometry_results(self, results: Tuple, phot_type: str, fltr: str) -> None:
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
        if phot_type in ['normal', 'optimal']:
            mjds, bdts, fluxes, flux_errors, detections = self._parse_photometry_results(results, phot_type)
        else:
            mjds, bdts, normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, detections = self._parse_photometry_results(results, phot_type)
        
        # save light curves
        for i in tqdm(range(len(self.catalogs[fltr])), disable=not self.verbose, desc=f"[OPTICAM] Saving {fltr} light curves"):
            if phot_type == 'normal':
                self._save_normal_light_curve(mjds, bdts, fluxes, flux_errors, fltr, i)
            elif phot_type == 'optimal':
                self._save_optimal_light_curve(mjds, bdts, fluxes, flux_errors, fltr, i)
            else:
                self._save_normal_light_curve(mjds, bdts, normal_fluxes, normal_flux_errors, fltr, i)
                self._save_optimal_light_curve(mjds, bdts, optimal_fluxes, optimal_flux_errors, fltr, i)
        
        # plot number of detections per source
        self._plot_number_of_detections_per_source(detections, fltr)

    def _save_normal_light_curve(self, mjds: List[float], bdts: List[float], fluxes: List[List[float]],
                                 flux_errors: List[List[float]], fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        mjds : List[float]
            The observation MJDs.
        bdts : List[float]
            The observation BDTs.
        fluxes : List[List[float]]
            The source fluxes.
        flux_errors : List[List[float]]
            The source flux errors.
        fltr : str
            The filter.
        source_index : int
            The source index, not to be confused with the source number. The source index is one less than the source
            number.
        """
        
        with open(self.out_directory + f"normal_light_curves/{fltr}_source_{source_index + 1}.csv", "w") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(["MJD", 'BDT', "flux", "flux_error"])
            
            # for each observation in which a source was detected
            for i in range(len(mjds)):
                try:
                    csvwriter.writerow([mjds[i], bdts[i], fluxes[i][source_index], flux_errors[i][source_index]])
                except TypeError:
                    continue
        
        df = pd.read_csv(self.out_directory + f"normal_light_curves/{fltr}_source_{source_index + 1}.csv")
        
        # reformat MJD to seconds from first observation
        df["time"] = df["MJD"] - self.t_ref
        df["time"] *= 86400
        
        fig, ax = plt.subplots(num=1, clear=True, tight_layout=True, figsize=(6.4, 4.8))
        
        ax.errorbar(df["time"].values, df["flux"].values, yerr=df["flux_error"].values, fmt="k.", ms=2, ecolor="grey",
                    elinewidth=1)
        ax.set_ylabel("Flux [counts]")
        ax.set_title(f"{fltr} Source {source_index + 1}")
        ax.set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
        
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        # save light curve plot to file
        fig.savefig(self.out_directory + f"normal_light_curves/{fltr}_source_{source_index + 1}.png")
        fig.clear()
        plt.close(fig)

    def _save_optimal_light_curve(self, mjds: List[float], bdts: List[float], fluxes: List[List[float]],
                                  flux_errors: List[List[float]], fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        mjds : List[float]
            The observation MJDs.
        bdts : List[float]
            The observation BDTs.
        fluxes : List[List[float]]
            The source fluxes.
        flux_errors : List[List[float]]
            The source flux errors.
        fltr : str
            The filter.
        source_index : int
            The source index, not to be confused with the source number. The source index is one less than the source
            number.
        """
        
        with open(self.out_directory + f"optimal_light_curves/{fltr}_source_{source_index + 1}.csv", "w") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(["MJD", 'BDT', "flux", "flux_error"])
            
            # for each observation in which a source was detected
            for i in range(len(mjds)):
                try:
                    csvwriter.writerow([mjds[i], bdts[i], fluxes[i][source_index], flux_errors[i][source_index]])
                except TypeError:
                    continue
        
        df = pd.read_csv(self.out_directory + f"optimal_light_curves/{fltr}_source_{source_index + 1}.csv")
        
        # reformat MJD to seconds from first observation
        df["time"] = df["MJD"] - self.t_ref
        df["time"] *= 86400
        
        # TODO: add local background axis
        
        fig, ax = plt.subplots(num=1, clear=True, tight_layout=True, figsize=(6.4, 4.8))
        
        ax.errorbar(df["time"].values, df["flux"].values, yerr=df["flux_error"].values, fmt="k.", ms=2, ecolor="grey",
                    elinewidth=1)
        ax.set_ylabel("Flux [counts]")
        ax.set_title(f"{fltr} Source {source_index + 1}")
        ax.set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
        
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        # save light curve plot to file
        fig.savefig(self.out_directory + f"optimal_light_curves/{fltr}_source_{source_index + 1}.png")
        fig.clear()
        plt.close(fig)

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
        
        aperture = EllipticalAperture(position, self.fwhm_scale * semimajor_sigma, self.fwhm_scale * semiminor_sigma,
                                      orientation)  # define aperture
        phot_table = aperture_photometry(data, aperture, error=error)  # perform aperture photometry
        
        if estimate_local_background:
            local_background_per_pixel, local_background_error_per_pixel = self.local_background(data, error, self.scale * semimajor_sigma, self.scale * semiminor_sigma, orientation, position)
            aperture_area = aperture.area_overlap(data)  # compute aperture area
            aperture_background, aperture_background_error = aperture_area * local_background_per_pixel, np.sqrt(aperture_area * local_background_error_per_pixel)  # compute aperture background
            
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
            local_background_per_pixel, local_background_error_per_pixel = self.local_background(data, error, self.scale * semimajor_sigma, self.scale * semiminor_sigma, orientation, position)
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
        catalog_position = (self.catalogs[fltr]["xcentroid"][source_index], self.catalogs[fltr]["ycentroid"][source_index])
        
        # if file is the reference image
        if file == self.camera_files[fltr][self.reference_indices[fltr]]:
            # use the catalog position as the initial position
            initial_position = catalog_position
        else:
            # use the transformed catalog position as the initial position
            initial_position = matrix_transform(catalog_position, self.transforms[file])[0]
        
        # get positions of sources
        positions = np.array([[file_tbl["xcentroid"][i], file_tbl["ycentroid"][i]] for i in range(len(file_tbl))])
        
        # get distances between sources and initial position
        distances = np.sqrt((positions[:, 0] - initial_position[0])**2 + (positions[:, 1] - initial_position[1])**2)
        
        # if the closest source is further than the specified tolerance
        if np.min(distances) > tolerance*np.sqrt(self.catalogs[fltr]["semimajor_sigma"][source_index].value**2 + self.catalogs[fltr]["semiminor_sigma"][source_index].value**2):
            raise ValueError(f"[OPTICAM] No source found close enough to source {source_index + 1} in {file}. Consider increasing the tolerance if the field is not too crowded or the alignments are poor.")
        else:
            # get the position of the closest source (assumed to be the source of interest)
            return positions[np.argmin(distances)]

    def _plot_number_of_detections_per_source(self, detections: NDArray, fltr: str) -> None:
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
        with open(self.out_directory + f"diag/{fltr}_observations.csv", "w") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(["source", "observations"])
            
            # for each source
            for i in range(len(self.catalogs[fltr])):
                csvwriter.writerow([i + 1, detections[i]])
        
        fig, ax = plt.subplots(tight_layout=True)
        
        ax.bar(np.arange(len(self.catalogs[fltr])) + 1, detections, color="none", edgecolor="black")
        ax.axhline(len(self.camera_files[fltr]), color="red", linestyle="--", lw=1)
        
        ax.set_xlabel("Source")
        ax.set_ylabel("Number of detections")
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        fig.savefig(self.out_directory + f"diag/{fltr}_observations.png")
        
        if self.show_plots:
            plt.show(fig)
        else:
            plt.close(fig)
