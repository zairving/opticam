import os
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from astropy.table import QTable
import json
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.visualization.mpl_normalize import simple_norm
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from astropy.time import Time
from photutils.segmentation import SourceCatalog, detect_threshold
from photutils.background import Background2D
from skimage.transform import estimate_transform, warp, matrix_transform, SimilarityTransform
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from multiprocessing import cpu_count
from functools import partial
from PIL import Image
from typing import List, Dict, Literal, Callable, Tuple
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ccdproc import cosmicray_lacosmic  # TODO: replace with astroscrappy to reduce dependencies?
import logging
import warnings
import pandas as pd

from opticam_new.helpers import log_binnings, log_filters, recursive_log
from opticam_new.helpers import bar_format, pixel_scales
from opticam_new.background import DefaultBackground
from opticam_new.finder import DefaultFinder
from opticam_new.correctors import FlatFieldCorrector
from opticam_new.photometers import BasePhotometer
from opticam_new.helpers import camel_to_snake, sort_filters


class Catalogue:
    """
    Create a catalogue of sources from OPTICAM data.
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
        background: None | Callable = None,
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
        background: Callable, optional
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
        
        ########################################### file paths ###########################################
        
        self.file_paths = []
        if self.data_directory is not None:
            file_names = sorted(os.listdir(self.data_directory))
            for file in file_names:
                if file.endswith('.fit') or file.endswith('.fits') or file.endswith('.fit.gz') or file.endswith('.fits.gz'):
                    self.file_paths.append(os.path.join(self.data_directory, file))
        else:
            if self.c1_directory is not None:
                file_names = sorted(os.listdir(self.c1_directory))
                for file in file_names:
                    if file.endswith('.fit') or file.endswith('.fits') or file.endswith('.fit.gz') or file.endswith('.fits.gz'):
                        self.file_paths.append(os.path.join(self.c1_directory, file))
            if self.c2_directory is not None:
                file_names = sorted(os.listdir(self.c2_directory))
                for file in file_names:
                    if file.endswith('.fit') or file.endswith('.fits') or file.endswith('.fit.gz') or file.endswith('.fits.gz'):
                        self.file_paths.append(os.path.join(self.c2_directory, file))
            if self.c3_directory is not None:
                file_names = sorted(os.listdir(self.c3_directory))
                for file in file_names:
                    if file.endswith('.fit') or file.endswith('.fits') or file.endswith('.fit.gz') or file.endswith('.fits.gz'):
                        self.file_paths.append(os.path.join(self.c3_directory, file))
        
        # list of files to ignore (e.g., if they are corrupted or do not comply with the FITS standard)
        self.ignored_files = []
        
        ########################################### scan data ###########################################
        
        self._scan_data_directory()  # scan data directory
        
        ########################################### catalogue colours ###########################################
        
        # define colours for circling sources in catalogues
        self.colours = list(mcolors.TABLEAU_COLORS.keys())
        self.colours.pop(self.colours.index("tab:brown"))
        self.colours.pop(self.colours.index("tab:gray"))
        self.colours.pop(self.colours.index("tab:purple"))
        self.colours.pop(self.colours.index("tab:blue"))
        
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
            self.logger.info(f'[OPTICAM] Using custom background estimator {background.__name__} with parameters {background.__dict__}.')
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
            self.logger.info(f'[OPTICAM] Using custom source finder {finder.__name__} with parameters {finder.__dict__}.')
        else:
            raise ValueError('[OPTICAM] finder must be a callable or None. If None, the default source finder is used.')
        
        ########################################### log input params ###########################################
        
        self._log_parameters()
        
        ########################################### misc attributes ###########################################
        
        self.transforms = {}  # define transforms as empty dictionary
        self.unaligned_files = []  # define unaligned files as empty list
        self.catalogues = {}  # define catalogues as empty dictionary
        self.psf_params = {}  # define PSF parameters as empty dictionary
        
        ########################################### read transforms ###########################################
        
        if os.path.isfile(os.path.join(self.out_directory, "cat/transforms.json")):
            with open(os.path.join(self.out_directory, "cat/transforms.json"), "r") as file:
                self.transforms.update(json.load(file))
            if self.verbose:
                self.logger.info("[OPTICAM] Read transforms from file.")
        
        ########################################### read catalogues ###########################################
        
        for fltr in list(self.camera_files.keys()):
            if os.path.isfile(os.path.join(self.out_directory, f"cat/{fltr}_catalogue.ecsv")):
                self.catalogues.update(
                    {
                        fltr: QTable.read(
                            os.path.join(self.out_directory, f"cat/{fltr}_catalogue.ecsv"),
                            format="ascii.ecsv",
                            )
                        }
                    )
                self._set_psf_params(fltr)
                if self.verbose:
                    print(f"[OPTICAM] Read {fltr} catalogue from file.")

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
        
        # scan files
        chunksize = max(1, len(self.file_paths) // 100)  # set chunksize to 1% of the number of files
        results = process_map(
            self._get_header_info,
            self.file_paths,
            max_workers=self.number_of_processors,
            disable=not self.verbose,
            desc="[OPTICAM] Scanning data directory",
            chunksize=chunksize,
            bar_format=bar_format,
            tqdm_class=tqdm)
        
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
        self.camera_files = sort_filters(self.camera_files)
        
        # sort files by time
        for key in list(self.camera_files.keys()):
            self.camera_files[key].sort(key=lambda x: self.bdts[x])
        
        self.t_ref = min(list(self.bdts.values()))  # get reference BDT
        
        # define middle image as reference image for each filter
        self.reference_indices = {}
        self.reference_files = {}
        for key in list(self.camera_files.keys()):
            self.reference_indices[key] = int(len(self.camera_files[key]) / 2)
            self.reference_files[key] = self.camera_files[key][self.reference_indices[key]]
        
        self.logger.info(f'[OPTICAM] Binning: {self.binning}')
        self.logger.info(f'[OPTICAM] Filters: {", ".join(list(self.camera_files.keys()))}')
        for fltr in list(self.camera_files.keys()):
            self.logger.info(f'[OPTICAM] {len(self.camera_files[fltr])} {fltr} images.')
        
        if self.verbose:
            print('[OPTICAM] Binning: ' + self.binning)
            print('[OPTICAM] Filters: ' + ', '.join(list(self.camera_files.keys())))
            for fltr in list(self.camera_files.keys()):
                print(f'[OPTICAM] {len(self.camera_files[fltr])} {fltr} images.')
        
        self._plot_time_between_files()

    def _get_header_info(self, file: str) -> Tuple[ArrayLike | None, str | None, str | None, float | None]:
        """
        Get the MJD, filter, binning, and gain from a file header.
        
        Parameters
        ----------
        file : str
            The file path.
        
        Returns
        -------
        Tuple[float, str, str, float]
            The BDT, filter, binning, and gain dictionaries.
        
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
                
                mjd = self._get_time(hdul, file)
                
                # separate files by filter
                fltr = hdul[0].header["FILTER"]
            
            try:
                # try to compute barycentric dynamical time
                coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                bdt = self._apply_barycentric_correction(mjd, coords)
            except Exception as e:
                bdt = mjd
                self.logger.info(f"[OPTICAM] Could not compute BDT for {file}: {e}. Using MJD instead.")
        except Exception as e:
            self.logger.info(f'[OPTICAM] Skipping file {file} because it could not be read: {e}')
            return None, None, None, None
        
        return bdt, fltr, binning, gain

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
        
        self.bdts = {}
        filters = {}
        binnings = {}
        self.gains = {}
        
        # unpack results
        raw_bdts, raw_filters, raw_binnings, raw_gains = zip(*results)
        
        # consolidate results
        for i in range(len(raw_bdts)):
            if raw_bdts[i] is not None:
                self.bdts.update({self.file_paths[i]: raw_bdts[i]})
                filters.update({self.file_paths[i]: raw_filters[i]})
                binnings.update({self.file_paths[i]: raw_binnings[i]})
                self.gains.update({self.file_paths[i]: raw_gains[i]})
            else:
                self.ignored_files.append(self.file_paths[i])
        
        # ensure there are no more than three filters
        unique_filters = np.unique(list(filters.values()))
        if unique_filters.size > 3:
            log_filters([file_path for file_path in self.file_paths if file_path not in self.ignored_files],
                        self.out_directory)
            raise ValueError(f"[OPTICAM] More than three filters found. Image filters have been logged to {os.path.join(self.out_directory, 'diag/filters.json')}.")
        
        # ensure there is at most one type of binning
        unique_binning = np.unique(list(binnings.values()))
        if len(unique_binning) > 1:
            log_binnings([file_path for file_path in self.file_paths if file_path not in self.ignored_files],
                         self.out_directory)
            raise ValueError(f"[OPTICAM] Inconsistent binning detected. All images must have the same binning. Image binnings have been logged to {os.path.join(self.out_directory, 'diag/binnings.json')}.")
        else:
            self.binning = unique_binning[0]
            self.binning_scale = int(self.binning[0])
        
        # check for large differences in time
        for fltr in unique_filters:
            bdts = np.array([self.bdts[file] for file in self.file_paths if file in filters and filters[file] == fltr])
            files = [file for file in self.file_paths if file in filters and filters[file] == fltr]
            t = bdts - np.min(bdts)
            dt = np.diff(t) * 86400
            if np.any(dt > 10 * np.median(dt)):
                indices = np.where(dt > 10 * np.median(dt))[0]
                for index in indices:
                    string = f"[OPTICAM] Large time gap detected between {files[index].split('/')[-1]} and {files[index + 1].split('/')[-1]} ({dt[index]:.3f} s compared to the median time difference of {np.median(dt):.3f} s). This may cause alignment issues. If so, consider moving all files after this gap to a separate directory."
                    self.logger.info(string)
                    warnings.warn(string)
        
        return filters

    def _log_parameters(self):
        """
        Log any and all object parameters to a JSON file.
        """
        
        # get parameters
        params = dict(recursive_log(self, max_depth=5))
        
        params.update({'filters': list(self.camera_files.keys())})
        
        # remove some parameters that are either already saved elsewhere or are not relevant
        params.pop('logger')  # redundant
        params.pop('bdts')  # redundant
        params.pop('gains')  # irrelevant
        params.pop('camera_files')  # redundant
        params.pop('colours')  # irrelevant
        params.pop('file_paths')  # redundant
        
        try:
            params.pop('transforms')  # redundant
        except KeyError:
            pass
        
        try:
            params.pop('unaligned_files')  # redundant
        except KeyError:
            pass
        
        try:
            params.pop('catalogues')  # redundant
        except KeyError:
            pass
        
        # sort parameters
        params = dict(sorted(params.items()))
        
        # write parameters to file
        with open(os.path.join(self.out_directory, "misc/input_parameters.json"), "w") as file:
            json.dump(params, file, indent=4)

    def _plot_time_between_files(self) -> None:
        """
        Plot the times between each file for each camera.
        
        Parameters
        ----------
        show : bool
            Whether to display the plot.
        """
        
        fig, axes = plt.subplots(nrows=3, ncols=len(self.camera_files), tight_layout=True,
                                 figsize=((2 * len(self.camera_files) / 3) * 6.4, 2 * 4.8), sharey='row')
        
        for fltr in list(self.camera_files.keys()):
            times = np.array([self.bdts[file] for file in self.camera_files[fltr]])
            times -= times.min()
            times *= 86400  # convert to seconds from first observation
            dt = np.diff(times)  # get time between files
            file_numbers = np.arange(2, len(times) + 1, 1)  # start from 2 because we are plotting the time between files
            
            bin_edges = np.arange(int(dt.min()), np.ceil(dt.max() + .2), .1)  # define bins with width 0.1 s
            
            if len(self.camera_files) == 1:
                axes[0].set_title(fltr)
                
                # cumulative plot of time between files
                axes[0].plot(file_numbers, np.cumsum(dt), "k-", lw=1)
                
                # time between each file
                axes[1].plot(file_numbers, dt, "k-", lw=1)
                
                axes[2].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
                axes[2].set_yscale("log")
                
                axes[0].set_ylabel("Cumulative time between files [s]")
                axes[0].set_xlabel("File number")
                
                axes[1].set_ylabel("Time between files [s]")
                axes[1].set_xlabel("File number")
                
                axes[2].set_xlabel("Time between files [s]")
            else:
                axes[0, list(self.camera_files.keys()).index(fltr)].set_title(fltr)
                
                # cumulative plot of time between files
                axes[0, list(self.camera_files.keys()).index(fltr)].plot(file_numbers, np.cumsum(dt), "k-", lw=1)
                
                # time between each file
                axes[1, list(self.camera_files.keys()).index(fltr)].plot(file_numbers, dt, "k-", lw=1)
                
                # histogram of time between files
                axes[2, list(self.camera_files.keys()).index(fltr)].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
                axes[2, list(self.camera_files.keys()).index(fltr)].set_yscale("log")
                
                axes[0, 0].set_ylabel("Cumulative time between files [s]")
                axes[1, 0].set_ylabel("Time between files [s]")
                
                for col in range(len(self.camera_files)):
                    axes[0, col].set_xlabel("File number")
                    axes[1, col].set_xlabel("File number")
                    axes[2, col].set_xlabel("Time between files [s]")
        
        for ax in axes.flatten():
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)

        fig.savefig(os.path.join(self.out_directory, "diag/header_times.png"))

        if self.show_plots:
            plt.show(fig)
        else:
            fig.clear()
            plt.close(fig)

    def _set_psf_params(self, fltr: str) -> None:
        """
        Set the PSF parameters for a given filter based on the catalogue data.
        
        Parameters
        ----------
        fltr : str
            The filter for which to set the PSF parameters.
        """
        
        self.psf_params[fltr] = {
            'semimajor_sigma': self.aperture_selector(self.catalogues[fltr]['semimajor_sigma'].value),
            'semiminor_sigma': self.aperture_selector(self.catalogues[fltr]['semiminor_sigma'].value),
            'orientation': self.aperture_selector(self.catalogues[fltr]['orientation'].value)
        }
        
        self.logger.info(f'[OPTICAM] {fltr} PSF parameters:')
        self.logger.info(f'[OPTICAM]    semimajor_sigma: {self.psf_params[fltr]["semimajor_sigma"]} binned pixels ({self.psf_params[fltr]["semimajor_sigma"] * self.binning_scale * self.rebin_factor * pixel_scales[fltr]} arcsec)')
        self.logger.info(f'[OPTICAM]    semiminor_sigma: {self.psf_params[fltr]["semiminor_sigma"]} binned pixels ({self.psf_params[fltr]["semiminor_sigma"] * self.binning_scale * self.rebin_factor * pixel_scales[fltr]} arcsec)')
        self.logger.info(f'[OPTICAM]    orientation: {self.psf_params[fltr]["orientation"]} degrees')

    def _get_data(self, file: str, return_error: bool = False) -> NDArray | Tuple[NDArray, NDArray]:
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
            error = np.sqrt(data * self.gains[file])
        
        if self.flat_corrector is not None:
            data = self.flat_corrector.correct(data, fltr)
            
            if return_error:
                error = self.flat_corrector.correct(error, fltr)
        
        # remove cosmic rays if required
        if self.remove_cosmic_rays:
            data = cosmicray_lacosmic(data, gain_apply=False)[0]
        
        if self.rebin_factor > 1:
            data = self._rebin_image(data, self.rebin_factor)
            
            if return_error:
                error = self._rebin_image(error, self.rebin_factor)
        
        if return_error:
            return data, error
        
        return data

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
        tbl = SourceCatalog(image_clean, cat, background=bkg.background).to_table()  # create catalogue of sources
        tbl.sort('segment_flux', reverse=True)  # sort catalogue by flux in descending order
        
        coords = np.array([tbl["xcentroid"], tbl["ycentroid"]]).T
        
        if away_from_edge:
            edge = 2 * self.background.box_size
            for coord in coords:
                if coord[0] < edge or coord[0] > image.shape[1] - edge or coord[1] < edge or coord[1] > image.shape[0] - edge:
                    coords = np.delete(coords, np.where(np.all(coords == coord, axis=1)), axis=0)
        
        if n_sources is not None:
            coords = coords[:n_sources]
        
        return coords


    def initialise(
        self,
        max_catalog_sources: int = 30,
        n_alignment_sources: int = 3,
        transform_type: Literal['euclidean', 'similarity', 'translation'] = 'translation',
        translation_limit: int | None = None,
        rotation_limit: int | None = None,
        scaling_limit: int | None = None,
        overwrite: bool = False,
        show_diagnostic_plots: bool = False,
        ) -> None:
        """
        Initialise the source catalogues for each camera. Some aspects of this method are parallelised for speed.
        
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
            Whether to overwrite existing catalogues, by default False.
        show_diagnostic_plots : bool, optional
            Whether to show diagnostic plots, by default False. Diagnostic plots are saved to out_directory, so this
            parameter only affects whether the plots are displayed in the console.
        """
        
        # if catalogues already exist, skip
        if os.path.isfile(os.path.join(self.out_directory, 'cat/catalogues.png')) and not overwrite:
            print('[OPTICAM] Catalogs already exist. To overwrite, set overwrite to True.')
            return
        
        if self.verbose:
            print('[OPTICAM] Initialising catalogues ...')
        
        # set some default transform limits if not provided
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
            
            reference_image = self._get_data(self.camera_files[fltr][self.reference_indices[fltr]])  # get reference image
            
            try:
                reference_coords = self._get_source_coords_from_image(reference_image, away_from_edge=True, n_sources=n_alignment_sources)  # get source coordinates in descending order of brightness
            except:
                self.logger.info(f'[OPTICAM] No sources detected in {fltr} reference image ({self.camera_files[fltr][self.reference_indices[fltr]]}). Reducing threshold or npixels in the source finder may help.')
                continue
            
            if len(reference_coords) < n_alignment_sources:
                self.logger.info(f'[OPTICAM] Not enough sources detected in {fltr} reference image ({self.camera_files[fltr][self.reference_indices[fltr]]}) for alignment. Reducing threshold and/or n_alignment_sources may help.')
                continue
            
            self.logger.info(f'[OPTICAM] {fltr} alignment source coordinates: {reference_coords}')
            
            # align and stack images in batches
            batches = np.array_split(self.camera_files[fltr], 100)  # split files into 1% batches
            results = process_map(
                partial(self._align_image,
                        reference_image=reference_image,
                        reference_coords=reference_coords,
                        transform_type=transform_type,
                        translation_limit=translation_limit,
                        rotation_limit=rotation_limit,
                        scaling_limit=scaling_limit),
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
                reference_image,
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
            
            tbl = SourceCatalog(stacked_image, segment_map).to_table()  # create catalogue of sources
            tbl.sort('segment_flux', reverse=True)
            tbl = tbl[:max_catalog_sources]  # limit catalogue to brightest max_catalog_sources sources
            
            # create catalogue of sources in stacked image and write to file
            self.catalogues.update({fltr: tbl})
            self.catalogues[fltr].write(
                os.path.join(self.out_directory, f"cat/{fltr}_catalogue.ecsv"),
                format="ascii.ecsv",
                overwrite=True,
                )
            
            # save stacked image to file
            np.savez_compressed(
                os.path.join(
                    self.out_directory,
                    f'cat/{fltr}_stacked_image.npz'),
                stacked_image=stacked_image,
                )
            
            self._set_psf_params(fltr)  # set PSF parameters for the filter
        
        self._plot_catalog(stacked_images)
        
        # diagnostic plots
        self._plot_backgrounds(background_median, background_rms, show_diagnostic_plots)  # plot background medians and RMSs
        self._plot_background_meshes(stacked_images, show_diagnostic_plots)  # plot background meshes
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

    def _align_image(
        self,
        batch: List[str],
        reference_image: NDArray,
        reference_coords: NDArray,
        transform_type: Literal['euclidean', 'similarity', 'translation'],
        translation_limit: int,
        rotation_limit: int,
        scaling_limit: int,
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
        
        stacked_image = np.zeros_like(reference_image)  # create empty stacked image
        transforms = {}
        background_medians = {}
        background_rmss = {}
        
        for file in batch:
            
            data = self._get_data(file)  # get image data
            
            bkg = self.background(data)  # get background
            background_median = bkg.background_median
            background_rms = bkg.background_rms_median
            
            data_clean = data - bkg.background  # remove background from image
            
            try:
                coords = self._get_source_coords_from_image(data, bkg)  # get source coordinates in descending order of brightness
            except:
                self.logger.info('[OPTICAM] No sources detected in ' + file + '. Reducing threshold or npixels in the source finder may help.')
                continue
            
            distance_matrix = cdist(reference_coords, coords)  # compute distance matrix
            try:
                reference_indices, indices = linear_sum_assignment(distance_matrix)  # solve assignment problem
            except:
                self.logger.info('[OPTICAM] Could not align ' + file + '. Reducing threshold and/or n_alignment_sources may help.')
                continue
            
            # compute transform
            if transform_type == 'translation':
                dx = np.mean(coords[indices, 0] - reference_coords[reference_indices, 0])
                dy = np.mean(coords[indices, 1] - reference_coords[reference_indices, 1])
                if abs(dx) < translation_limit and abs(dy) < translation_limit:
                    transform = SimilarityTransform(translation=[dx, dy])
                else:
                    self.logger.info(f'[OPTICAM] File {file} exceeded translation limit. Translation limit is {translation_limit:.1f}, but translation was ({dx:.1f}, {dy:.1f}).')
                    continue
            else:
                transform = estimate_transform(transform_type, reference_coords[reference_indices], coords[indices])
                # TODO: implement transform constraints
            
            transforms[file] = transform.params.tolist()  # save transform parameters
            background_medians[file] = background_median  # save background median
            background_rmss[file] = background_rms  # save background RMS
            
            # transform and stack image
            stacked_image += warp(data_clean, transform.inverse, output_shape=reference_image.shape, order=3,
                                    mode='constant', cval=np.nanmedian(data), clip=True, preserve_range=True)
        
        return stacked_image, transforms, background_medians, background_rmss

    def _parse_alignment_results(self, results, fltr: str, reference_image) -> Tuple[NDArray, Dict[str, float],
                                                                                     Dict[str, float]]:
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

    def _plot_catalog(self, stacked_images: Dict[str, NDArray]) -> None:
        """
        Plot the source catalogues on top of the stacked images
        
        Parameters
        ----------
        stacked_images : Dict[str, NDArray]
            The stacked images for each camera.
        """
        
        fig, ax = plt.subplots(ncols=len(self.catalogues), tight_layout=True, figsize=(len(stacked_images) * 5, 5))
        
        if len(self.catalogues) == 1:
            ax = [ax]
        
        for i, fltr in enumerate(list(self.catalogues.keys())):
            
            plot_image = np.clip(stacked_images[fltr], 0, None)  # clip negative values to zero for better visualisation
            
            # plot stacked image
            ax[i].imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                            norm=simple_norm(plot_image, stretch="log"))
            
            # get aperture radius
            radius = 5 * self.aperture_selector(self.catalogues[fltr]["semimajor_sigma"].value)
            
            for j in range(len(self.catalogues[fltr])):
                # label sources
                ax[i].add_patch(Circle(xy=(self.catalogues[fltr]["xcentroid"][j],
                                            self.catalogues[fltr]["ycentroid"][j]),
                                        radius=radius, edgecolor=self.colours[j % len(self.colours)], 
                                        facecolor="none", lw=1))
                ax[i].text(self.catalogues[fltr]["xcentroid"][j] + 1.05*radius,
                            self.catalogues[fltr]["ycentroid"][j] + 1.05*radius, j + 1, 
                            color=self.colours[j % len(self.colours)])
                
                # label plot
                ax[i].set_title(fltr)
                ax[i].set_xlabel("X")
                ax[i].set_ylabel("Y")

        fig.savefig(os.path.join(self.out_directory, "cat/catalogues.png"))

        if self.show_plots:
            plt.show(fig)
        else:
            fig.clear()
            plt.close(fig)

    def _plot_background_meshes(self, stacked_images: Dict[str, NDArray], show: bool) -> None:
        """
        Plot the background meshes on top of the catalogue images.
        
        Parameters
        ----------
        stacked_images : Dict[str, NDArray]
            The stacked images for each camera.
        show : bool
            Whether to display the plot.
        """
        
        fig, ax = plt.subplots(ncols=len(self.catalogues), tight_layout=True, figsize=(len(self.catalogues) * 5, 5))
        
        for i, fltr in enumerate(list(self.catalogues.keys())):
            
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

    def _plot_backgrounds(
        self,
        background_median: Dict[str, Dict[str, NDArray]],
        background_rms: Dict[str, Dict[str, NDArray]],
        show: bool,
        ) -> None:
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
        
        fig, axs = plt.subplots(nrows=2, ncols=len(self.catalogues), tight_layout=True, figsize=((2 * len(self.catalogues) / 3) * 6.4, 2 * 4.8), sharex='col')
        
        # for each camera
        for fltr in list(self.catalogues.keys()):
            
            files = self.camera_files[fltr]  # get files for camera
            
            # skip cameras with no images
            if len(files) == 0:
                continue
            
            # get values from background_median and background_rms dicts
            backgrounds = list(background_median[fltr].values())
            rmss = list(background_rms[fltr].values())
            
            # match times to background_median and background_rms keys
            bdts = np.array([self.bdts[file] for file in files if file in background_median[fltr]])
            plot_times = (bdts - self.t_ref) * 86400  # convert time to seconds from first observation
            
            if len(self.catalogues) == 1:
                axs[0].set_title(fltr)
                axs[0].plot(plot_times, backgrounds, "k.", ms=2)
                axs[1].plot(plot_times, rmss, "k.", ms=2)
                
                axs[1].set_xlabel(f"Time from TDB {bdts.min():.4f} [s]")
                axs[0].set_ylabel("Median background RMS")
                axs[1].set_ylabel("Median background")
            else:
                # plot background
                axs[0, list(self.catalogues.keys()).index(fltr)].set_title(fltr)
                axs[0, list(self.catalogues.keys()).index(fltr)].plot(plot_times, backgrounds, "k.", ms=2)
                axs[1, list(self.catalogues.keys()).index(fltr)].plot(plot_times, rmss, "k.", ms=2)
                
                for col in range(len(self.catalogues)):
                    axs[1, col].set_xlabel(f"Time from TDB {bdts.min():.4f} [s]")
                
                axs[0, 0].set_ylabel("Median background")
                axs[1, 0].set_ylabel("Median background RMS")
            
            # write background to file
            pd.DataFrame({
                'TDB': bdts,
                'RMS': backgrounds,
                'median': rmss
            }).to_csv(os.path.join(self.out_directory, f'diag/{fltr}_background.csv'), index=False)

        for ax in axs.flatten():
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
        # save plot
        fig.savefig(os.path.join(self.out_directory, "diag/background.png"))
        
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
        for fltr in list(self.catalogues.keys()):
            
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
        
        data = self._get_data(file)
        
        file_name = file.split('/')[-1].split(".")[0]
        
        bkg = self.background(data)
        clean_data = data - bkg.background
        
        # clip negative values to zero for better visualisation
        plot_image = np.clip(clean_data, 0, None)
        
        fig, ax = plt.subplots(num=1, clear=True, tight_layout=True)
        
        ax.imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                norm=simple_norm(plot_image, stretch="log"))
        
        # for each source
        for i in range(len(self.catalogues[fltr])):
            
            source_position = (self.catalogues[fltr]["xcentroid"][i], self.catalogues[fltr]["ycentroid"][i])
            
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
            
            radius = 5 * self.aperture_selector(self.catalogues[fltr]["semimajor_sigma"].value)
            
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
        Perform photometry on the catalogues using the provided photometer.
        
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
        for fltr in self.catalogues.keys():
            
            source_coords = np.array([self.catalogues[fltr]["xcentroid"].value,
                                      self.catalogues[fltr]["ycentroid"].value]).T
            
            chunk_size = max(1, len(self.camera_files[fltr]) // 100)
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
                chunksize=chunk_size,
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
            
            # for each source in the catalogue
            for i in tqdm(
                range(len(self.catalogues[fltr])),
                disable=not self.verbose,
                desc=f"[OPTICAM] Saving {fltr} photometry results",
                ):
                
                # unpack results for ith source
                source_results = {}
                for key, values in photometry_results.items():
                    
                    # time is a special case since it is already a single column
                    if key == 'TDB':
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
        
        image, error = self._get_data(file, return_error=True)  # get image data and error
        
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
        results['TDB'] = self.bdts[file]  # add time of observation
        
        return results


    @staticmethod
    def _get_time(
        hdul,
        file: str,
        ) -> float:
        """
        Parse the time from the header of a FITS file.
        
        Parameters
        ----------
        hdul
            The FITS file.
        file : str
            The path to the file.
        
        Returns
        -------
        float
            The time of the observation in MJD.
        
        Raises
        ------
        KeyError
            If neither 'GPSTIME' nor 'UT' keys are found in the header.
        ValueError
            If the time cannot be parsed from the header.
        """
        
        # parse file time
        if "GPSTIME" in hdul[0].header.keys():
            gpstime = hdul[0].header["GPSTIME"]
            split_gpstime = gpstime.split(" ")
            date = split_gpstime[0]  # get date
            time = split_gpstime[1].split(".")[0]  # get time (ignoring decimal seconds)
            mjd = Time(date + "T" + time, format="fits").mjd
        elif "UT" in hdul[0].header.keys():
            try:
                mjd = Time(hdul[0].header["UT"].replace(" ", "T"), format="fits").mjd
            except:
                try:
                    date = hdul[0].header['DATE-OBS']
                    time = hdul[0].header['UT'].split('.')[0]
                    mjd = Time(date + 'T' + time, format='fits').mjd
                except:
                    raise ValueError('Could not parse time from ' + file + ' header.')
        else:
            raise KeyError(f"[OPTICAM] Could not find GPSTIME or UT key in {file} header.")
        
        return mjd

    def identify_gaps(
        self,
        files: List[str],
        ) -> None:
        """
        Identify gaps in the observation sequence and logs them to log_dir/diag/gaps.txt.
        
        Parameters
        ----------
        files : List[str]
            The list of files for a single filter.
        """
        
        file_times = {}
        
        for file in tqdm(files, desc='[OPTICAM] Identifying gaps'):
            with fits.open(file) as hdul:
                file_times[file] = self._get_time(hdul, file)
        
        sorted_files = dict(sorted(file_times.items(), key=lambda x: x[1]))
        times = np.array(sorted_files.values()).flatten()
        diffs = np.diff(times)
        median_exposure_time = np.median(diffs)
        
        gaps = np.where(diffs > 2 * median_exposure_time)[0]
        
        if len(gaps) > 0:
            print(f'[OPTICAM] Found {len(gaps)} gaps in the observation sequence.')
            with open(os.path.join(self.out_directory, 'diag/gaps.txt'), 'w') as file:
                file.write(f"Median exposure time: {median_exposure_time} d\n")
                for gap in gaps:
                    file.write(f"Gap between {list(sorted_files.keys())[gap]} and {list(sorted_files.keys())[gap + 1]}: {diffs[gap]} d\n")
        else:
            print('[OPTICAM] No gaps found in the observation sequence.')

    @staticmethod
    def _apply_barycentric_correction(
        original_times: float | NDArray,
        coords: SkyCoord,
        ) -> float | NDArray:
        """
        Apply barycentric corrections to a time array.
        
        Parameters
        ----------
        times : float | NDArray
            The time(s) to correct.
        coords : SkyCoord
            The coordinates of the source.
        
        Returns
        -------
        float | NDArray
            The corrected time(s).
        """
        
        # OPTICAM location
        observer_coords = EarthLocation.from_geodetic(lon=-115.463611*u.deg, lat=31.044167*u.deg, height=2790*u.m)
        
        # format the times
        times = Time(original_times, format='mjd', scale='utc', location=observer_coords)
        
        # compute light travel time to barycentre
        ltt_bary = times.light_travel_time(coords)
        
        return (times.tdb + ltt_bary).value

    @staticmethod
    def _rebin_image(
        image: NDArray,
        factor: int,
        ) -> NDArray:
        """
        Rebin an image by a given factor in both dimensions.
        
        Parameters
        ----------
        image : NDArray
            The image to rebin.
        factor : int
            The factor to rebin by.
        
        Returns
        -------
        NDArray
            The rebinned image.
        """
        
        if image.shape[0] % factor != 0 or image.shape[1] % factor != 0:
            raise ValueError(f'[OPTICAM] The dimensions of the input data must be divisible by the rebinning factor. Got shape {image.shape} and factor {factor}.')
        
        # reshape the array to efficiently rebin
        shape = (image.shape[0] // factor, factor, image.shape[1] // factor, factor)
        reshaped_data = image.reshape(shape)
        
        # rebin image by summing over the new axes
        return reshaped_data.sum(axis=(1, 3))