from weakref import ref
from cycler import K
from tqdm import tqdm
from astropy.table import QTable
import json
import numpy as np
from astropy.time import Time
import os
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.visualization.mpl_normalize import simple_norm
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.segmentation import SourceCatalog, detect_threshold
from photutils.aperture import ApertureStats, aperture_photometry, CircularAperture, CircularAnnulus, EllipticalAperture, EllipticalAnnulus
from photutils.background import Background2D
from photutils.utils import calc_total_error
from skimage.transform import estimate_transform, warp, matrix_transform, SimilarityTransform
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.colors as mcolors
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image
from typing import List, Dict, Literal, Callable, Tuple, Union
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import csv
import warnings
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from ccdproc import cosmicray_lacosmic

from opticam_new.helpers import get_data, log_binnings, log_filters, default_aperture_selector, apply_barycentric_correction, clip_extended_sources
from opticam_new.background import Background
from opticam_new.local_background import EllipticalLocalBackground
from opticam_new.finder import CrowdedFinder, Finder

try:
    os.environ['OMP_NUM_THREADS'] = '1'  # set number of threads to 1 for better multiprocessing performance
except:
    pass

# TODO: check input and output signatures
# TODO: add FWHM column to catalog tables
# TODO: optimal photometry with annulus


class Reducer:
    """
    Helper class for reducing OPTICAM data.
    """
    
    def __init__(
        self,
        data_directory: str,
        out_directory: str,
        threshold: float = 5,
        background: Callable = None,
        local_background: Callable = None,
        finder: Union[Literal['crowded', 'default'], Callable] = 'default',
        aperture_selector: Callable = None,
        scale: float = 5,
        remove_cosmic_rays: bool = True,
        number_of_processors: int = int(cpu_count()/2),
        show_plots: bool = True,
        verbose: bool = True
        ) -> None:
        """
        Helper class for reducing OPTICAM data.
        
        Parameters
        ----------
        data_directory: str
            The path to the directory containing the data.
        out_directory: str
            The path to the directory to save the output files.
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
        
        self.data_directory = data_directory
        if not self.data_directory[-1].endswith("/"):
            self.data_directory += "/"
        
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
        
        # create subdirectories
        if not os.path.isdir(self.out_directory + "cat"):
            os.makedirs(self.out_directory + "cat")
        if not os.path.isdir(self.out_directory + "diag"):
            os.makedirs(self.out_directory + "diag")
        if not os.path.isdir(self.out_directory + "misc"):
            os.makedirs(self.out_directory + "misc")
        
        # set parameters
        self.fwhm_scale = 2 * np.sqrt(2 * np.log(2))  # FWHM scale factor
        self.aperture_selector = default_aperture_selector if aperture_selector is None else aperture_selector
        self.scale = scale
        self.threshold = threshold
        self.remove_cosmic_rays = remove_cosmic_rays
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        self.file_names = sorted(os.listdir(self.data_directory))  # get list of file names
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
        
        # get input parameters and write to file
        param_dict = {
            "aperture selector": self.aperture_selector.__name__,
            "aperture scale": scale,
            "threshold": threshold,
        }
        param_dict.update({"number of files": len(self.file_names)})
        param_dict.update({f"number of {fltr} files": len(self.camera_files[fltr]) for fltr in list(self.camera_files.keys())})
        with open(self.out_directory + "misc/reducer_input.json", "w") as file:
            json.dump(param_dict, file, indent=4)
        
        # define background calculator and write input parameters to file
        if background is None:
            self.background = Background(box_size=int(64/self.binning_scale))
        else:
            self.background = background
        # TODO: improve parameter logging to file
        try:
            with open(self.out_directory + "misc/background_input.json", "w") as file:
                json.dump(self.background.get_input_dict(), file, indent=4)
        except:
            warnings.warn("[OPTICAM] Could not write background input parameters to file. It's a good idea to add a get_input_dict() method to your background estimator for reproducability (see the background tutorial).")
        
        if local_background is None:
            self.local_background = EllipticalLocalBackground()
        else:
            self.local_background = local_background
        
        # define source finder and write input parameters to file
        if finder == 'default':
            self.finder = Finder(npixels=int(128 / (self.binning_scale * 2)), border_width=int(64 / (self.binning_scale * 2)))
        elif finder == 'crowded':
            self.finder = CrowdedFinder(npixels=int(128 / (self.binning_scale * 2)), border_width=int(64 / (self.binning_scale * 2)))
        elif callable(finder):
            self.finder = finder
        else:
            raise ValueError("[OPTICAM] Source finder must be 'default', 'crowded', or a callable.")
        
        try:
            with open(self.out_directory + "misc/finder_input.json", "w") as file:
                json.dump(self.finder.get_input_dict(), file, indent=4)
        except:
            warnings.warn("[OPTICAM] Could not write finder input parameters to file. It's a good idea to add a get_input_dict() method to your source finder for reproducability (see the source finder tutorial).")
        
        self.transforms = {}  # define transforms as empty dictionary
        self.unaligned_files = []  # define unaligned files as empty list
        self.catalogs = {}  # define catalogs as empty dictionary
        
        # try to load transforms from file
        try:
            with open(self.out_directory + "cat/transforms.json", "r") as file:
                self.transforms.update(json.load(file))
            if self.verbose:
                print("[OPTICAM] Read transforms from file.")
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
    
    def _scan_data_directory(self, batch_size: int = None) -> None:
        """
        Scan the data directory for files and extract the MJD, filter, binning, and gain from the file headers.
        
        Parameters
        ----------
        batch_size : int, optional
            The number of images to process in each batch, by default the batch size is based on the number of images, 
            the number of unique filters, and the number of processors. For the best performance, the number of batches
            should not be less than the number of processors.
        
        Raises
        ------
        ValueError
            If more than 3 filters are found.
        ValueError
            If the binning is not consistent.
        """
        
        if batch_size is None:
            batch_size = 1 + int(len(self.file_names)/self.number_of_processors)
        
        batches = [self.file_names[i:i + batch_size] for i in range(0, len(self.file_names), batch_size)]
        
        if self.verbose:
            print("[OPTICAM] Scanning files ...")
        
        self.camera_files = {}  # filter : [files]
        
        if self.verbose:
            # scan files in parallel with progress bar
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(self._scan_batch, batches), total=len(batches)))
        else:
            # scan files in parallel without progress bar
            with Pool(self.number_of_processors) as pool:
                results = pool.map(self._scan_batch, batches)
        
        # unpack results
        self.mjds, self.bdts, filters, self.gains = self._parse_batch_scanning_results(results)
        
        # for each unique filter
        for fltr in np.unique(list(filters.values())):
            
            self.camera_files.update({fltr + '-band': []})  # prepare dictionary entry
            
            # for each file
            for file in self.file_names:
                # if the file filter matches the current filter
                if filters[file] == fltr:
                    self.camera_files[fltr + '-band'].append(file)  # add file name to dict list
        
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
        
        with open(self.out_directory + "misc/earliest_observation_time.txt", "w") as file:
            file.write(str(self.t_ref))
        
        if self.verbose:
            print('[OPTICAM] Done.')
            print('[OPTICAM] Binning: ' + self.binning)
            print('[OPTICAM] Filters: ' + ', '.join(list(self.camera_files.keys())))
    
    def _scan_batch(self, batch: List[str]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[str]], Dict[str, List[float]], Dict[str, List[float]]]:
        """
        Get the MJD, filter, binning, and gain from a batch of file headers.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        
        Returns
        -------
        Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[str]], Dict[str, List[float]], Dict[str, List[float]]]
            The MJD, BDT, filter, binning, and gain dictionaries.
        
        Raises
        ------
        KeyError
            If the file header does not contain the required keys.
        """
        
        mjds = {}
        bdts = {}
        filters = {}
        binnings = {}
        gains = {}
        
        for file in batch:
            with fits.open(self.data_directory + file) as hdul:
                binnings[file] = hdul[0].header["BINNING"]
                gains[file] = hdul[0].header["GAIN"]
                
                try:
                    ra = hdul[0].header["RA"]
                    dec = hdul[0].header["DEC"]
                except:
                    pass
                
                # parse file time
                if "GPSTIME" in hdul[0].header.keys():
                    gpstime = hdul[0].header["GPSTIME"]
                    split_gpstime = gpstime.split(" ")
                    date = split_gpstime[0]  # get date
                    time = split_gpstime[1].split(".")[0]  # get time (ignoring decimal seconds)
                    mjds[file] = Time(date + "T" + time, format="fits").mjd
                elif "UT" in hdul[0].header.keys():
                    try:
                        mjds[file] = Time(hdul[0].header["UT"].replace(" ", "T"), format="fits").mjd
                    except:
                        try:
                            date = hdul[0].header['DATE-OBS']
                            time = hdul[0].header['UT'].split('.')[0]
                            mjds[file] = Time(date + 'T' + time, format='fits').mjd
                        except:
                            raise ValueError('Could not parse time from ' + file + ' header.')
                else:
                    raise KeyError(f"[OPTICAM] Could not find GPSTIME or UT key in {file} header.")
                
                # separate files by filter
                filters[file] = hdul[0].header["FILTER"]
            
            try:
                # try to compute barycentric dynamical time
                coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
                bdts[file] = apply_barycentric_correction(mjds[file], coords)
            except:
                bdts[file] = mjds[file]
            
        return mjds, bdts, filters, binnings, gains
    
    def _parse_batch_scanning_results(self, results: List[Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[str]], Dict[str, List[float]], Dict[str, List[float]]]]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[str]], Dict[str, List[float]]]:
        """
        Parse the results of a batch of file scanning.
        
        Parameters
        ----------
        results : 
            The batch results.  
        
        Returns
        -------
        
            The MJDs, BDTs, filters, binnings, and gains.
        
        Raises
        ------
        ValueError
            If more than 3 filters are found.
        ValueError
            If the binning is not consistent.
        """
        
        mjds = {}
        bdts = {}
        filters = {}
        binnings = {}
        gains = {}
        
        # unpack results
        batch_mjds, batch_bdts, batch_filters, batch_binnings, batch_gains = zip(*results)
        
        # consolidate results
        for i in range(len(batch_mjds)):
            mjds.update(batch_mjds[i])
            bdts.update(batch_bdts[i])
            filters.update(batch_filters[i])
            binnings.update(batch_binnings[i])
            gains.update(batch_gains[i])
        
        # ensure there are no more than three filters
        unique_filters = np.unique(list(filters.values()))
        if unique_filters.size > 3:
            log_filters(self.data_directory, self.out_directory)
            raise ValueError("[OPTICAM] More than 3 filters found. Image filters have been logged to {self.out_directory}misc/filters.json.")
        else:
            with open(self.out_directory + "misc/filters.txt", "w") as file:
                for fltr in unique_filters:
                    file.write(f"{fltr}-band\n")
        
        # ensure there is at most one type of binning
        unique_binning = np.unique(list(binnings.values()))
        if len(unique_binning) > 1:
            log_binnings(self.data_directory, self.out_directory)
            raise ValueError(f"[OPTICAM] Inconsistent binning detected. All images must have the same binning. Image binnings have been logged to {self.out_directory}diag/binnings.json.")
        else:
            self.binning = unique_binning[0]
            self.binning_scale = int(self.binning[0])
        
        return mjds, bdts, filters, gains




    def get_source_coords_from_image(self, image: ArrayLike, bkg: Background2D = None) -> ArrayLike:
        """
        Get an array of source coordinates from an image in descending order of source brightness.
        
        Parameters
        ----------
        image : ArrayLike
            The image from which to extract source coordinates.
        bkg : Background2D, optional
            The background of the image, by default None. If None, the background is estimated from the image. Including
            this parameter can prevent the background from being estimated multiple times.
        
        Returns
        -------
        ArrayLike
            The source coordinates in descending order of brightness.
        """
        
        if bkg is None:
            bkg = self.background(image)  # get background
        
        image_clean = image - bkg.background  # remove background from image
        
        cat = self.finder(image_clean, self.threshold*bkg.background_rms)  # find sources in background-subtracted image
        tbl = SourceCatalog(image_clean, cat, background=bkg.background).to_table()  # create catalog of sources
        tbl = clip_extended_sources(tbl)
        tbl.sort('segment_flux', reverse=True)  # sort catalog by flux in descending order
        
        # return source coordinates in descending order of brightness
        return np.array([tbl["xcentroid"], tbl["ycentroid"]]).T




    def initialise_catalogs(self, n_alignment_sources: int = 3,
                            transform_type: Literal['euclidean', 'similarity', 'translation'] = 'translation',
                            batch_size: int = None, overwrite: bool = False) -> None:
        """
        Initialise the source catalogs for each camera. Some aspects of this method are parallelised for speed.
        
        Parameters
        ----------
        n_alignment_sources : int, optional
            The number of sources to use for image alignment, by default 3. Must be >= 3. The brightest
            n_alignment_sources sources are used for image alignment.
        batch_size : int, optional
            The number of images to process in each batch, by default the batch size is based on the number of images, 
            the number of unique filters, and the number of processors. For the best performance, the number of batches
            should not be less than the number of processors.
        overwrite : bool, optional
            Whether to overwrite existing catalogs, by default False.
        """
        
        # if catalogs already exist, skip
        if os.path.isfile(self.out_directory + 'cat/catalogs.png') and not overwrite:
            print('[OPTICAM] Catalogs already exist. To overwrite, set overwrite to True.')
            return
        
        # automatically determine a suitable batch size
        if batch_size is None:
            infer_batch_size = True
        
        if self.verbose:
            print('[OPTICAM] Initialising catalogs ...')
        
        background_median = {}
        background_rms = {}
        
        stacked_images = {}
        
        # for each camera
        for fltr in self.camera_files.keys():
            
            # if no images found for camera, skip
            if len(self.camera_files[fltr]) == 0:
                continue
            
            reference_image = get_data(self.data_directory + self.camera_files[fltr][self.reference_indices[fltr]])  # get reference image
            reference_coords = self.get_source_coords_from_image(reference_image)  # get source coordinates in descending order of brightness
            
            if len(reference_coords) < n_alignment_sources:
                print('[OPTICAM] Not enough sources detected in ' + fltr + ' for image alignment. Consider reducing threshold and/or n_alignment_sources.')
                continue
            
            if infer_batch_size:
                batch_size = 1 + int(len(self.camera_files[fltr])/self.number_of_processors)
            
            # split files into batches for improved performance
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            # align and stack images
            if self.verbose:
                print('[OPTICAM] Aligning and stacking ' + fltr + ' images in batches ...')
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(self._align_and_stack_image_batch, reference_image=reference_image, reference_coords=reference_coords, n_sources=n_alignment_sources, transform_type=transform_type), batches), total=len(batches)))
            else:
                with Pool(self.number_of_processors) as pool:
                    results = pool.map(partial(self._align_and_stack_image_batch, reference_image=reference_image, reference_coords=reference_coords, n_sources=n_alignment_sources, transform_type=transform_type), batches)
            
            # parse batch results
            stacked_image, background_median[fltr], background_rms[fltr] = self._parse_batch_alignment_and_stacking_results(results, reference_image.copy())
            
            try:
                threshold = detect_threshold(stacked_image, nsigma=self.threshold, sigma_clip=SigmaClip(sigma=3, maxiters=10))  # estimate threshold
            except:
                if self.verbose:
                    print('[OPTICAM] Unable to estimate source detection threshold for ' + fltr + '.')
                continue
            
            try:
                # identify sources in stacked image
                segment_map = self.finder(stacked_image, self.threshold*threshold)
            except:
                if self.verbose:
                    print('[OPTICAM] No sources detected in the stacked ' + fltr + ' image.')
                continue
            
            # save stacked image and its background
            stacked_images[fltr] = stacked_image
            
            tbl = SourceCatalog(stacked_image, segment_map).to_table()  # create catalog of sources
            tbl = clip_extended_sources(tbl)  # clip extended sources
            
            # create catalog of sources in stacked image and write to file
            self.catalogs.update({fltr: tbl})
            self.catalogs[fltr].write(self.out_directory + f"cat/{fltr}_catalog.ecsv", format="ascii.ecsv",
                                            overwrite=True)
        
        # render plots
        self._plot_catalog(stacked_images)
        # self._show_source_psfs(stacked_images)
        # self._plot_background_meshes(stacked_images, stacked_backgrounds)
        self._plot_time_between_files()  # plot time between observations
        self._plot_backgrounds(background_median, background_rms)  # plot background medians and RMSs
        # self._plot_stacked_backgrounds(stacked_images)
        
        # save transforms to file
        with open(self.out_directory + "cat/transforms.json", "w") as file:
            json.dump(self.transforms, file, indent=4)
        
        # write unaligned files to file
        if len(self.unaligned_files) > 0:
            with open(self.out_directory + "diag/unaligned_files.txt", "w") as unaligned_file:
                for file in self.unaligned_files:
                    unaligned_file.write(file + "\n")
    
    def _align_and_stack_image_batch(self, batch: List[str], reference_image: NDArray, reference_coords, n_sources, transform_type) -> Tuple[Dict[str, List], List, ArrayLike, List, List]:
        """
        Align and stack a batch of images.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        reference_image : ArrayLike
            The reference image to align the batch images to.
        
        Returns
        -------
        Tuple[Dict[str, List], List, ArrayLike, List, List]
            The transforms, unaligned files, stacked image, background medians, and background RMSs.
        """
        
        transforms = {}
        unaligned_files = []
        background_median = []
        background_rms = []
        
        stacked_image = np.zeros_like(reference_image)
        
        for file in batch:
            data = get_data(self.data_directory + file)  # get image data
            
            if self.remove_cosmic_rays:
                data = cosmicray_lacosmic(data, gain_apply=False)[0]
            
            bkg = self.background(data)  # get background
            background_median.append(bkg.background_median)
            background_rms.append(bkg.background_rms_median)
            
            data_clean = data - bkg.background  # remove background from image
            
            try:
                coords = self.get_source_coords_from_image(data_clean)  # get source coordinates in descending order of brightness
            except:
                print('[OPTICAM] No sources detected in ' + file + '.')
                unaligned_files.append(file)
                continue
            
            distance_matrix = cdist(reference_coords, coords)  # compute distance matrix
            reference_indices, indices = linear_sum_assignment(distance_matrix)  # solve assignment problem
            
            try:
                reference_indices = reference_indices[:n_sources]
                indices = indices[:n_sources]
            except:
                print('[OPTICAM] Could not align ' + file + '. Consider reducing threshold and/or n_alignment_sources.')
                unaligned_files.append(file)  # store unaligned file in list
                continue
            
            
            # compute transform
            if transform_type == 'translation':
                dx = np.mean(coords[indices, 0] - reference_coords[reference_indices, 0])
                dy = np.mean(coords[indices, 1] - reference_coords[reference_indices, 1])
                # r = np.sqrt(dx**2 + dy**2)
                transform = SimilarityTransform(translation=[dx, dy])
            else:
                transform = estimate_transform(transform_type, reference_coords[reference_indices], coords[indices])
            
            transforms.update({file: transform.params.tolist()})  # store transform in dictionary
            stacked_image += warp(data_clean, transform.inverse, output_shape=stacked_image.shape, order=3, mode='constant', cval=np.nanmedian(data), clip=True, preserve_range=True)  # align and stack image
        
        return transforms, unaligned_files, stacked_image, background_median, background_rms
    
    def _parse_batch_alignment_and_stacking_results(self, results: List[Tuple[Dict[str, List], List, ArrayLike, List, List]], stacked_image: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Parse the results of a batch of image alignment and stacking.
        
        Parameters
        ----------
        results : List[Tuple[Dict[str, List], List, ArrayLike, List, List]]
            The batch results.
        stacked_image : ArrayLike
            The stacked image onto which the batch images are stacked.
        
        Returns
        -------
        Tuple[ArrayLike, ArrayLike, ArrayLike]
            The stacked image, background medians, and background RMSs.
        """
        
        # unpack results
        batch_transforms, batch_unaligned_files, batch_stacked_images, batch_background_medians, batch_background_rmss = zip(*results)
        
        stacked_image += np.sum(batch_stacked_images, axis=0)  # stack batch images
        transforms = {k: v for d in batch_transforms for k, v in d.items()}  # combine batch transforms
        unaligned_files = [file for batch in batch_unaligned_files for file in batch]  # combine batch unaligned files
        background_median = np.array([median for batch in batch_background_medians for median in batch]).flatten()  # combine batch background medians
        background_rms = np.array([rms for batch in batch_background_rmss for rms in batch]).flatten()  # combine batch background RMSs
        
        self.transforms.update(transforms)  # update transforms
        self.unaligned_files += unaligned_files  # update unaligned files
        
        if self.verbose:
            print(f"[OPTICAM] Done. {len(unaligned_files)} image(s) could not be aligned.")
        
        return stacked_image, background_median, background_rms
    
    def _plot_catalog(self, stacked_images: Dict[str, ArrayLike]) -> None:
        
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
            plt.close(fig)
    
    def _plot_background_meshes(self, stacked_images: Dict[str, ArrayLike],
                                stacked_backgrounds: Dict[str, ArrayLike]) -> None:
        
        fig, ax = plt.subplots(ncols=len(self.catalogs), tight_layout=True, figsize=(len(self.catalogs) * 5, 5))
        
        for i, fltr in enumerate(list(self.catalogs.keys())):
            
            plot_image = np.clip(stacked_images[fltr], 0, None)
            
            try:
                # plot background mesh
                ax[i].imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                             norm=simple_norm(plot_image, stretch="log"))
                stacked_backgrounds[fltr].plot_meshes(ax=ax[i], outlines=True, marker='.', color='cyan', alpha=0.3)
                
                #label plot
                ax[i].set_title(fltr)
                ax[i].set_xlabel("X")
                ax[i].set_ylabel("Y")
            except:
                # plot background mesh
                ax.imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                          norm=simple_norm(plot_image, stretch="log"))
                stacked_backgrounds[fltr].plot_meshes(ax=ax, outlines=True, marker='.', color='cyan', alpha=0.3)
                
                # label plot
                ax.set_title(fltr)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
        
        fig.savefig(self.out_directory + "diag/background_meshes.png")
        
        if self.show_plots:
            plt.show(fig)
        else:
            plt.close(fig)
    
    def _plot_time_between_files(self) -> None:
        """
        Plot the times between each file for each camera.
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
        
        if self.show_plots:
            plt.show(fig)
        else:
            plt.close(fig)
    
    def _plot_backgrounds(self, background_median: Dict[str, List], background_rms: Dict[str, List]) -> None:
        """
        Plot the time-varying background for each camera.
        
        Parameters
        ----------
        background_median : Dict[str, List]
            The median background for each camera.
        background_rms : Dict[str, List]
            The background RMS for each camera.
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
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    
    def _show_source_psfs(self, stacked_images: Dict[str, ArrayLike]) -> None:
        
        # for each filter
        for fltr in list(self.catalogs.keys()):
            
            if not os.path.isdir(self.out_directory + 'diag/' + fltr + '_apertures'):
                os.mkdir(self.out_directory + 'diag/' + fltr + '_apertures')
            
            # for each source
            for source in range(len(self.catalogs[fltr])):
                
                #isolate region around source
                x = self.catalogs[fltr]["xcentroid"][source]
                x_shift = x - int(x)
                y = self.catalogs[fltr]["ycentroid"][source]
                y_shift = y - int(y)
                size = 10*self.catalogs[fltr]["semimajor_sigma"][source].value
                
                row_mask = np.zeros(stacked_images[fltr].shape[0], dtype=bool)
                row_mask[int(y - size):int(y + size)] = True
                
                col_mask = np.zeros(stacked_images[fltr].shape[1], dtype=bool)
                col_mask[int(x - size):int(x + size)] = True
                
                source_img = stacked_images[fltr][row_mask][:, col_mask]
                img_x, img_y = source_img.shape[0] / 2 + x_shift, source_img.shape[1] / 2 + y_shift
                
                norm = simple_norm(stacked_images[fltr], stretch="log")
                
                self._get_flux_profile(source_img, img_x, img_y, self.catalogs[fltr]['semimajor_sigma'][source].value,
                                       self.catalogs[fltr]['semiminor_sigma'][source].value,
                                       self.catalogs[fltr]['orientation'][source].value, fltr, source, norm)
                
                #show PSF
                fig, axs = plt.subplots(ncols=2, tight_layout=True, figsize=(10, 5))
                
                axs[0].imshow(np.clip(source_img, 0, None), origin="lower",
                              cmap="Greys_r", interpolation="nearest", norm=norm)
                
                axs[0].add_patch(Ellipse(xy=(img_x, img_y),
                                         width=self.catalogs[fltr]['semiminor_sigma'][source].value,
                                         height=self.catalogs[fltr]['semimajor_sigma'][source].value,
                                         angle=self.catalogs[fltr]['orientation'][source].value / np.pi * 180,
                                         edgecolor='red', facecolor='none', lw=1))
                
                axs[1].hist(source_img.flatten(), bins='auto', histtype='step', color='black', lw=1)
                
                axs1_ = axs[1].twinx()
                
                aperture_mask = np.zeros_like(source_img, dtype=bool)
                aperture_mask = EllipticalAperture([img_x, img_y], self.catalogs[fltr]['semimajor_sigma'][source].value,
                                                   self.catalogs[fltr]['semimajor_sigma'][source].value,
                                                   self.catalogs[fltr]['orientation'][source].value).to_mask(method='center')
                
                axs1_.hist(aperture_mask.get_values(source_img).flatten(), bins='auto', histtype='step', color='red',
                           lw=1, ls='--')
                
                axs[0].set_title(f"{fltr} source {source + 1}")
                axs[0].set_xlabel("X")
                axs[0].set_ylabel("Y")
                
                axs[1].set_yscale('log')
                axs[1].set_xlabel("Flux")
                
                fig.savefig(self.out_directory + f"diag/{fltr}_apertures/source_{source + 1}_psf.png")
                
                plt.close(fig)
    
    def _get_flux_profile(self, img, x_centroid, y_centroid, semimajor_sigma, semiminor_sigma, orientation, fltr,
                          source, norm):
        
        scale = self.fwhm_scale * semimajor_sigma  # get PSF FWHM
        semimajor_sigma /= 2 * scale  # normalise to 1 pixel 'diameter'
        semiminor_sigma /= 2 * scale  # normalise to 1 pixel 'diameter'
        
        major, minor, circle = [], [], []
        major_alt, minor_alt, circle_alt = [], [], []
        
        semimajor_in = 0
        semimajor_out = semimajor_sigma
        semiminor_in = 0
        semiminor_out = semiminor_sigma
        
        while semimajor_out < img.shape[0] / 2:
            major.append(self._get_flux(img, (x_centroid, y_centroid), semimajor_in, semimajor_out, semiminor_out, semiminor_in, orientation))  # ellipse parallel to major axis
            minor.append(self._get_flux(img, (x_centroid, y_centroid), semimajor_in, semimajor_out, semiminor_out, semiminor_in, orientation + np.pi / 2))  # ellipse perpendicular to major axis
            circle.append(self._get_flux(img, (x_centroid, y_centroid), semimajor_in, semimajor_out, semimajor_out, semimajor_in, 0))  # circle
            
            major_alt.append(self._get_flux(img, (x_centroid, y_centroid), 0, semimajor_out, semiminor_out, 0, orientation))
            minor_alt.append(self._get_flux(img, (x_centroid, y_centroid), 0, semimajor_out, semiminor_out, 0, orientation + np.pi / 2))
            circle_alt.append(self._get_flux(img, (x_centroid, y_centroid), 0, semimajor_out, semimajor_out, 0, 0))
            
            semimajor_in = semimajor_out
            semimajor_out += semimajor_sigma
            semiminor_in = semiminor_out
            semiminor_out += semiminor_sigma
        
        fig, axs = plt.subplots(ncols=2, tight_layout=True, figsize=(10, 5))
        
        axs[0].imshow(np.clip(img, 0, None), origin="lower", cmap="Greys_r", interpolation="nearest", 
                      norm=norm)
        
        axs[0].add_patch(Ellipse(xy=(x_centroid, y_centroid), width=semiminor_sigma * 2 * scale, height=semimajor_sigma * 2 * scale,
                                 angle=orientation / np.pi * 180, edgecolor='red', facecolor='none', lw=1))
        
        axs[0].add_patch(Ellipse(xy=(x_centroid, y_centroid), width=semiminor_sigma * 2 * scale, height=semimajor_sigma * 2 * scale,
                                 angle=90 + orientation / np.pi * 180, edgecolor='blue', facecolor='none', lw=1, ls='--'))
        
        axs[0].add_patch(Ellipse(xy=(x_centroid, y_centroid), width=semimajor_sigma * 2 * scale, height=semimajor_sigma * 2 * scale,
                                 angle=0, edgecolor='green', facecolor='none', lw=1, ls=':'))
        
        axs[0].set_ylabel("Y")
        axs[0].set_xlabel("X")
        
        axs[1].plot(np.arange(1, len(major) + 1), major, "rx-", lw=1, label="Parallel")
        axs[1].plot(np.arange(1, len(minor) + 1), minor, "b+--", lw=1, label="Perpendicular")
        axs[1].plot(np.arange(1, len(circle) + 1), circle, "g1-.", lw=1, label="Circle")
        axs[1].axvline(semimajor_sigma * 2 * scale, color='black', ls='--', lw=1, label='PSF FWHM')
        
        axs[1].legend()
        axs[1].set_xlabel('Radius/semi-major length [pixels]')
        axs[1].set_ylabel('Flux')
        axs[1].set_title(f"{fltr} source {source + 1}")
        axs[1].minorticks_on()
        axs[1].tick_params(which='both', direction='in', top=True, right=True)
        
        fig.savefig(self.out_directory + f"diag/{fltr}_apertures/source_{source + 1}_flux_profile.png", dpi=300)
        plt.close(fig)
        
        
        fig, axs = plt.subplots(ncols=2, tight_layout=True, figsize=(10, 5))
        
        axs[0].imshow(np.clip(img, 0, None), origin="lower", cmap="Greys_r", interpolation="nearest", 
                      norm=norm)
        
        axs[0].add_patch(Ellipse(xy=(x_centroid, y_centroid), width=semiminor_sigma * 2 * scale, height=semimajor_sigma * 2 * scale,
                                 angle=orientation / np.pi * 180, edgecolor='red', facecolor='none', lw=1))
        
        axs[0].add_patch(Ellipse(xy=(x_centroid, y_centroid), width=semiminor_sigma * 2 * scale, height=semimajor_sigma * 2 * scale,
                                 angle=90 + orientation / np.pi * 180, edgecolor='blue', facecolor='none', lw=1, ls='--'))
        
        axs[0].add_patch(Ellipse(xy=(x_centroid, y_centroid), width=semimajor_sigma * 2 * scale, height=semimajor_sigma * 2 * scale,
                                 angle=0, edgecolor='green', facecolor='none', lw=1, ls=':'))
        
        axs[0].set_ylabel("Y")
        axs[0].set_xlabel("X")
        
        axs[1].plot(np.arange(1, len(major_alt) + 1), major_alt, "rx-", lw=1, label="Parallel")
        axs[1].plot(np.arange(1, len(minor_alt) + 1), minor_alt, "b+--", lw=1, label="Perpendicular")
        axs[1].plot(np.arange(1, len(circle_alt) + 1), circle_alt, "g1-.", lw=1, label="Circle")
        axs[1].axvline(semimajor_sigma * 2 * scale, color='black', ls='--', lw=1, label='PSF FWHM')
        
        axs[1].legend()
        axs[1].set_xlabel('Radius/semi-major length [pixels]')
        axs[1].set_ylabel('Flux')
        axs[1].set_title(f"{fltr} source {source + 1}")
        axs[1].minorticks_on()
        axs[1].tick_params(which='both', direction='in', top=True, right=True)
        
        fig.savefig(self.out_directory + f"diag/{fltr}_apertures/source_{source + 1}_flux_sum_profile.png")
        plt.close(fig)
    
    def _get_flux(self, data, position, a_in, a_out, b_out, b_in, theta):
        
        if a_in == 0:
            aperture = EllipticalAperture(position, a_out, b_out, theta)
        else:
            aperture = EllipticalAnnulus(position, a_in, a_out, b_out, b_in, theta)
        
        aperstats = ApertureStats(data, aperture)
        
        return aperstats.sum
    
    def _plot_stacked_backgrounds(self, stacked_images: Dict[str, ArrayLike]) -> None:
        
        fig, axs = plt.subplots(nrows=len(self.catalogs), tight_layout=True, figsize=(6.4, 4.8 * len(self.catalogs)))
        
        for i, fltr in enumerate(list(self.catalogs.keys())):
            
            plot_image = SigmaClip(3, maxiters=10)(stacked_images[fltr], axis=1, masked=False)
            
            axs[i].imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                          norm=simple_norm(plot_image, stretch="log"))
            
            axs[i].set_title(fltr)
            axs[i].set_xlabel("X")
            axs[i].set_ylabel("Y")
        
        fig.savefig(self.out_directory + "cat/stacked_backgrounds.png")
        
        if self.show_plots:
            plt.show(fig)
        else:
            plt.close(fig)

    
    def _background(self):
        
        batch_size = 1 + int(len(self.file_names) / (len(self.catalogs) * self.number_of_processors))
        
        fig, ax = plt.subplots(ncols=len(self.catalogs), tight_layout=True, figsize=(len(self.catalogs) * 5, 5))
        
        stacked_images = []
        
        for fltr in list(self.catalogs.keys()):
            
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            reference_image = get_data(self.data_directory + self.reference_files[fltr])
            
            for i in  range(10):
                median = np.median(reference_image)
                std = np.std(reference_image)
                temp = (reference_image - median) / std
                outliers = np.abs(temp) > 3
                reference_image[outliers] = median
            
            stacked_images.append(reference_image)
            
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self.temp, reference_image=get_data(self.data_directory + self.reference_files[fltr]), fltr=fltr), batches), total=len(batches)))
            
            for result in results:
                stacked_images[-1] += result
        
        for i, fltr in enumerate(list(self.catalogs.keys())):
            
            ax[i].imshow(stacked_images[i], origin="lower", cmap="Greys_r", interpolation="nearest",
                        norm=simple_norm(stacked_images[i], stretch="sqrt"))
            
            ax[i].set_title(fltr)
            ax[i].set_xlabel("X")
            ax[i].set_ylabel("Y")
        
        fig.savefig(self.out_directory + "cat/stacked_backgrounds.png")
    
    def temp(self, batch, reference_image, fltr):
        
        stacked_image = np.zeros_like(reference_image)
        
        for file in batch:
            
            try:
                itransform = SimilarityTransform(self.transforms[file]).inverse
            except KeyError:
                continue
            
            image = get_data(self.data_directory + file)
            
            for i in range(10):
                median = np.median(image)
                std = np.std(image)
                temp = (image - median) / std
                outliers = np.abs(temp) > 3
                image[outliers] = median
            
            stacked_image += warp(image, itransform, output_shape=image.shape, order=3, mode='constant', cval=np.nanmedian(image), clip=True, preserve_range=True)
        
        return stacked_image




    def create_gifs(self, keep_frames: bool = True, batch_size: int = None, overwrite: bool = False) -> None:
        """
        Create alignment gifs for each camera. Some aspects of this method are parallelised for speed. The frames are 
        saved in out_directory/diag/*-band_gif_frames and the GIFs are saved in out_directory/cat.
        
        Parameters
        ----------
        keep_frames : bool, optional
            Whether to save the GIF frames in out_directory/diag, by default True. If False, the frames will be deleted
            after the GIF is saved.
        batch_size : int, optional
            The number of images to process in each batch, by default the batch size is based on the number of images, 
            the number of unique filters, and the number of processors. For the best performance, the number of batches
            should not be less than the number of processors.
        overwrite : bool, optional
            Whether to overwrite existing GIFs, by default False.
        """
        
        if batch_size is None:
            n_catalog_files = sum([len(self.camera_files[fltr]) for fltr in list(self.catalogs.keys())])
            batch_size = 1 + int(n_catalog_files/(len(self.catalogs)*self.number_of_processors))
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            elif os.path.exists(self.out_directory + f"cat/{fltr}_images.gif"):
                print(f"[OPTICAM] {fltr} GIF already exists. To overwrite, set overwrite to True.")
                continue
            
            # create gif frames directory if it does not exist
            if not os.path.isdir(self.out_directory + f"diag/{fltr}_gif_frames"):
                os.mkdir(self.out_directory + f"diag/{fltr}_gif_frames")
            
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            if self.verbose:
                # create gif frames in parallel with progress bar
                print(f"[OPTICAM] Creating {fltr} GIF frames ...")
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(self._create_gif_frames, fltr=fltr), batches), total=len(batches)))
                print(f"[OPTICAM] Done.")
            else:
                # create gif frames in parallel without progress bar
                with Pool(self.number_of_processors) as pool:
                    results = pool.map(partial(self._create_gif_frames, fltr=fltr), batches)
            
            # save GIF
            self._compile_gif(fltr, keep_frames)
    
    def _create_gif_frames(self, batch: List[str], fltr: str) -> None:
        """
        Create a gif frames from a batch of images and save it to the out_directory.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        fltr : str
            The filter.
        """
        
        for file in batch:
            data = get_data(self.data_directory + file)
            
            bkg = self.background(data)
            clean_data = data - bkg.background
            
            # clip negative values to zero for better visualisation
            plot_image = np.clip(clean_data, 0, None)
            
            fig, ax = plt.subplots(tight_layout=True)  # set figure number to 999 to avoid conflict with other figures
            
            ax.imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                    norm=simple_norm(plot_image, stretch="log"))
            
            # for each source
            for i in range(len(self.catalogs[fltr])):
                
                source_position = (self.catalogs[fltr]["xcentroid"][i], self.catalogs[fltr]["ycentroid"][i])
                
                if file == self.reference_files[fltr]:
                    aperture_position = source_position
                    title = f"{file} (reference)"
                    colour = "blue"
                else:
                    try:
                        aperture_position = matrix_transform(source_position, self.transforms[file])[0]
                        title = f"{file} (aligned)"
                        colour = "black"
                    except:
                        aperture_position = source_position
                        title = f"{file} (unaligned)"
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
            
            fig.savefig(self.out_directory + 'diag/' + fltr + '_gif_frames/' + file.split(".")[0] + '.png')
            plt.close(fig)
    
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
        
        if self.verbose:
            print(f"[OPTICAM] Saving GIF (this can take some time) ...")
        
        # load frames
        frames = []
        for file in self.camera_files[fltr]:
            try:
                frames.append(Image.open(self.out_directory + 'diag/' + fltr + '_gif_frames/' + file.split(".")[0] + '.png'))
            except:
                pass
        
        # save gif
        frames[0].save(self.out_directory + 'cat/' + fltr + '_images.gif', format='GIF', append_images=frames[1:], 
                       save_all=True, duration=200, loop=0)
        del frames  # delete frames after gif is saved to clear memory
        
        # delete frames after gif is saved
        if not keep_frames:
            for file in os.listdir(self.out_directory + f"diag/{fltr}_gif_frames"):
                os.remove(self.out_directory + f"diag/{fltr}_gif_frames/{file}")
        
        if self.verbose:
            print("[OPTICAM] Done.")




    def background_gif(self, keep_frames=True, batch_size=None, overwrite=False):
            """
            Create alignment gifs for each camera. Some aspects of this method are parallelised for speed. The frames are 
            saved in out_directory/diag/*-band_gif_frames and the GIFs are saved in out_directory/cat.
            
            Parameters
            ----------
            keep_frames : bool, optional
                Whether to save the GIF frames in out_directory/diag, by default True. If False, the frames will be deleted
                after the GIF is saved.
            batch_size : int, optional
                The number of images to process in each batch, by default the batch size is based on the number of images, 
                the number of unique filters, and the number of processors. For the best performance, the number of batches
                should not be less than the number of processors.
            overwrite : bool, optional
                Whether to overwrite existing GIFs, by default False.
            """
            
            if batch_size is None:
                n_catalog_files = sum([len(self.camera_files[fltr]) for fltr in list(self.catalogs.keys())])
                batch_size = 1 + int(n_catalog_files/(len(self.catalogs)*self.number_of_processors))
            
            # for each camera
            for fltr in list(self.catalogs.keys()):
                
                if len(self.camera_files[fltr]) == 0:
                    continue
                
                if os.path.exists(self.out_directory + f"cat/{fltr}_background.gif") and not overwrite:
                    print(f"[OPTICAM] {fltr} background GIF already exists. To overwrite, set overwrite to True.")
                    continue
                
                # create gif frames directory if it does not exist
                if not os.path.isdir(self.out_directory + f"diag/{fltr}_background_gif_frames"):
                    os.mkdir(self.out_directory + f"diag/{fltr}_background_gif_frames")
                
                batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
                
                if self.verbose:
                    # create gif frames in parallel with progress bar
                    print(f"[OPTICAM] Creating {fltr} background GIF frames ...")
                    with Pool(self.number_of_processors) as pool:
                        results = list(tqdm(pool.imap(partial(self._create_background_gif_frames, fltr=fltr), batches), total=len(batches)))
                    print(f"[OPTICAM] Done.")
                else:
                    # create gif frames in parallel without progress bar
                    with Pool(self.number_of_processors) as pool:
                        results = pool.map(partial(self._create_background_gif_frames, fltr=fltr), batches)
                
                # save GIF
                self._compile_background_gif(fltr, keep_frames)
    
    def _create_background_gif_frames(self, batch: List[str], fltr: str) -> None:
        
        for file in batch:
            data = get_data(self.data_directory + file)
            
            clipped_data = SigmaClip(3, maxiters=10)(data, axis=1, masked=False)
            
            fig, ax = plt.subplots(tight_layout=True)
            
            ax.imshow(clipped_data, origin="lower", cmap="Greys", interpolation="nearest",
                      norm=simple_norm(clipped_data, stretch="log"))
            
            ax.set_title(file)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            fig.savefig(self.out_directory + 'diag/' + fltr + '_background_gif_frames/' + file.split(".")[0] + '.png')
            plt.close(fig)
    
    def _compile_background_gif(self, fltr: str, keep_frames: bool) -> None:
        
        if self.verbose:
            print(f"[OPTICAM] Saving background GIF (this can take some time) ...")
        
        # load frames
        frames = []
        for file in self.camera_files[fltr]:
            try:
                frames.append(Image.open(self.out_directory + 'diag/' + fltr + '_background_gif_frames/' + file.split(".")[0] + '.png'))
            except:
                pass
        
        # save gif
        frames[0].save(self.out_directory + 'cat/' + fltr + '_background.gif', format='GIF', append_images=frames[1:], 
                       save_all=True, duration=200, loop=0)
        del frames




    def forced_photometry(self, phot_type: Literal["aperture", "annulus", "both"] = "both",
                          batch_size: int = None, remove_cosmic_rays: bool = False, overwrite: bool = False) -> None:
        """
        Perform forced photometry on the images in out_directory to extract source fluxes.
        
        Parameters
        ----------
        phot_type : Literal["aperture", "annulus", "both"], optional
            The type of photometry to perform, by default "both". If "aperture", only aperture photometry is performed.
            If "annulus", only annulus photometry is performed. If "both", both aperture and annulus photometry are
            performed simultaneously (this is more efficient that performing both separately since it only opens the
            file once).
        batch_size : int, optional
            The number of images to process in each batch, by default the batch size is based on the number of images, 
            the number of unique filters, and the number of processors. For the best performance, the number of batches
            should not be less than the number of processors.
        remove_cosmic_rays : bool, optional
            Whether to remove cosmic rays from the images before performing photometry, by default False. Removing
            cosmic rays can reduce the number of outliers that appear in the resulting light curves, but doing so can
            significantly increase the processing time.
        overwrite : bool, optional
            Whether to overwrite existing light curves, by default False.
        
        Raises
        ------
        ValueError
            If phot_type is not recognised.
        """
        n_catalog_files = sum([len(self.camera_files[fltr]) for fltr in list(self.catalogs.keys())])
        batch_size = 1 + int(n_catalog_files/(len(self.catalogs)*self.number_of_processors))
        
        # determine which photometry function to use
        if phot_type == "aperture":
            self._extract_aperture_light_curves(batch_size, remove_cosmic_rays, overwrite)
        elif phot_type == "annulus":
            self._extract_annulus_light_curves(batch_size, remove_cosmic_rays, overwrite)
        elif phot_type == "both":
            self._extract_aperture_and_annulus_light_curves(batch_size, remove_cosmic_rays, overwrite)
        else:
            raise ValueError(f"[OPTICAM] Photometry type {phot_type} not recognised.")
    
    def _extract_aperture_light_curves(self, batch_size: int, remove_cosmic_rays: bool,
                                       overwrite: bool) -> None:
        """
        Perform forced simple aperture photometry on all the images in out_directory.
        
        batch_size : int
            The number of images to process in each batch.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        overwrite : bool
            Whether to overwrite existing light curves.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"aperture_light_curves"):
            os.mkdir(self.out_directory + f"aperture_light_curves")
        
        if self.verbose:
            print(f"[OPTICAM] Extracting aperture fluxes ...")
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            if os.path.isfile(self.out_directory + f"aperture_light_curves/{fltr}_source_1.csv") and not overwrite:
                print(f'[OPTICAM] {fltr} aperture light curves already exist. To overwrite, set overwrite to True.')
                continue
            
            try:
                # get aperture radius
                radius = self.scale*self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            except:
                # skip cameras with no sources
                continue
            
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            if self.verbose:
                print(f"[OPTICAM] Processing {fltr} files ...")
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(self._extract_aperture_fluxes_from_batch, fltr=fltr, radius=radius, remove_cosmic_rays=remove_cosmic_rays), batches), total=len(batches)))
                print("[OPTICAM] Done.")
            else:
                with Pool(self.number_of_processors) as pool:
                    results = pool.map(partial(self._extract_aperture_fluxes_from_batch, fltr=fltr, radius=radius, remove_cosmic_rays=remove_cosmic_rays), batches)
            
            # parse results
            fluxes, flux_errors, flags = self._parse_batch_forced_aperture_extration_results(results)
            
            mjds = [self.mjds[file] for file in self.camera_files[fltr]]
            bdts = [self.bdts[file] for file in self.camera_files[fltr]]
            
            if self.verbose:
                # save light curves with progress bar
                print("[OPTICAM] Saving light curves ...")
                for i in tqdm(range(len(self.catalogs[fltr]))):
                    self._save_aperture_light_curve(mjds, bdts, fluxes, flux_errors, flags, fltr, i)
                print("[OPTICAM] Done.")
            else:
                # save light curves without progress bar
                for i in range(len(self.catalogs[fltr])):
                    self._save_aperture_light_curve(mjds, bdts, fluxes, flux_errors, flags, fltr, i)
    
    def _extract_aperture_fluxes_from_batch(self, batch: List[str], fltr: str, radius: float, remove_cosmic_rays: bool) -> Tuple[List, List, str]:
        """
        Perform forced simple aperture photometry on a single image.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        fltr : str
            The filter.
        radius : float
            The aperture radius.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        
        Returns
        -------
        Tuple[List, List, str]
            The source fluxes, source flux errors, and image quality flag.
        """
        
        batch_fluxes, batch_flux_errors, flags = [], [], []
        
        for file in batch:
            
            fluxes, flux_errors = [], []
            
            # if file is the reference image
            if file == self.camera_files[fltr][self.reference_indices[fltr]]:
                flags.append("A")
            # else if file does not have a transform
            elif file not in self.transforms.keys():
                flags.append("B")
            # if file is not the reference image and has a transform
            else:
                flags.append("A")
                transform = self.transforms[file]
            
            clean_data, error = self._get_background_subtracted_image_and_error(file, remove_cosmic_rays)
            
            # for each source
            for i in range(len(self.catalogs[fltr])):
                
                catalog_position = (self.catalogs[fltr]["xcentroid"][i], self.catalogs[fltr]["ycentroid"][i])
                
                # try to transform source position
                try:
                    position = matrix_transform(catalog_position, transform)[0]
                # if transform fails, use catalog position
                except:
                    position = catalog_position
                
                # define aperture
                flux, flux_error = self._compute_aperture_flux(clean_data, error, position, radius)
                fluxes.append(flux)
                flux_errors.append(flux_error)
            
            batch_fluxes.append(fluxes)
            batch_flux_errors.append(flux_errors)
        
        return batch_fluxes, batch_flux_errors, flags
    
    def _get_background_subtracted_image_and_error(self, file: str, remove_cosmic_rays: bool) -> Tuple[ArrayLike, ArrayLike]:
        """
        Get the background subtracted image and error for a given file.
        
        Parameters
        ----------
        file : str
            The name of the file.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        
        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            The background subtracted image and its error.
        """
        
        data = get_data(self.data_directory + file)
        
        if remove_cosmic_rays:
            data = cosmicray_lacosmic(data, gain_apply=False)[0]
        
        bkg = self.background(data)
        clean_data = data - bkg.background
        error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])
        
        return clean_data, error
    
    @staticmethod
    def _compute_aperture_flux(clean_data: ArrayLike, error: ArrayLike, position: ArrayLike, radius: float) -> Tuple[float, float]:
        """
        Compute the flux and error for a given aperture position and radius.
        
        Parameters
        ----------
        clean_data : ArrayLike
            The background subtracted image.
        error : ArrayLike
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
    
    def _parse_batch_forced_aperture_extration_results(self, results: List[Tuple[List, List, str]]) -> Tuple[List, List, List]:
        """
        Parse the results from aperture photometry.
        
        Parameters
        ----------
        results : Tuple[List, List, str]
            The results from aperture photometry.
        
        Returns
        -------
        Tuple[List, List, List]
            The source fluxes, source flux errors, and image quality flags.
        """
        
        batch_aperture_fluxes, batch_aperture_flux_errors, batch_flags = zip(*results)
        
        aperture_fluxes, aperture_flux_errors = [], []
        flags = []
        
        for i in range(len(batch_aperture_fluxes)):
            aperture_fluxes += batch_aperture_fluxes[i]
            aperture_flux_errors += batch_aperture_flux_errors[i]
            flags += batch_flags[i]
        
        return aperture_fluxes, aperture_flux_errors, flags
    
    def _save_aperture_light_curve(self, mjds: ArrayLike, bdts: ArrayLike, fluxes: ArrayLike, flux_errors: ArrayLike,
                                   flags: ArrayLike, fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        mjds : ArrayLike
            The observation MJDs.
        bdts : ArrayLike
            The observation BDTs.
        fluxes : ArrayLike
            The source fluxes.
        flux_errors : ArrayLike
            The source flux errors.
        flags : ArrayLike
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
        
        fig, ax = plt.subplots(tight_layout=True, figsize=(6.4, 4.8))
        
        ax.errorbar(df["time"].values[aligned_mask], df["flux"].values[aligned_mask], yerr=df["flux_error"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
        ax.errorbar(df["time"].values[~aligned_mask], df["flux"].values[~aligned_mask], yerr=df["flux_error"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
        ax.set_ylabel("Flux [counts]")
        ax.set_title(f"{fltr} Source {source_index + 1}")
        ax.set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
        
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        # save light curve plot to file
        fig.savefig(self.out_directory + f"aperture_light_curves/{fltr}_source_{source_index + 1}.png")
        
        plt.close(fig)
    
    def _extract_annulus_light_curves(self,  batch_size: int, remove_cosmic_rays: bool, overwrite: bool) -> None:
        """
        Perform forced aperture photometry with local background subtractions on all the images in out_directory.
        
        Parameters
        ----------
        batch_size : int
            The number of images to process in each batch.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        overwrite : bool
            Whether to overwrite existing light curves.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"annulus_light_curves"):
            os.mkdir(self.out_directory + f"annulus_light_curves")
        
        if self.verbose:
            print(f"[OPTICAM] Extracting annulus fluxes ...")
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            if os.path.isfile(self.out_directory + f"annulus_light_curves/{fltr}_source_1.csv") and not overwrite:
                print(f'[OPTICAM] {fltr} annulus light curves already exist. To overwrite, set overwrite to True.')
                continue
            
            try:
                # get aperture radius
                radius = self.scale*self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            except:
                # skip cameras with no sources
                continue
            
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            if self.verbose:
                # perform annulus photometry in parallel with progress bar
                print(f"[OPTICAM] Processing {fltr} files ...")
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(self._extract_annulus_fluxes_from_batch, fltr=fltr, radius=radius, remove_cosmic_rays=remove_cosmic_rays), batches), total=len(batches)))
                print("[OPTICAM] Done.")
            else:
                # perform annulus photometry in parallel without progress bar
                with Pool(self.number_of_processors) as pool:
                    results = pool.map(partial(self._extract_annulus_fluxes_from_batch, fltr=fltr, radius=radius, remove_cosmic_rays=remove_cosmic_rays), batches)
            
            # unpack results
            fluxes, flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = self._parse_batch_forced_annulus_extraction_results(results)
            
            mjds = [self.mjds[file] for file in self.camera_files[fltr]]
            bdts = [self.bdts[file] for file in self.camera_files[fltr]]
            
            if self.verbose:
                # save light curves with progress bar
                print("[OPTICAM] Saving light curves ...")
                for i in range(len(self.catalogs[fltr])):
                    self._save_annulus_light_curve(mjds, bdts, fluxes, flux_errors, local_backgrounds,
                                                local_background_errors, local_backgrounds_per_pixel,
                                                local_background_errors_per_pixel, flags, fltr, i)
                print("[OPTICAM] Done.")
            else:
                for i in range(len(self.catalogs[fltr])):
                    self._save_annulus_light_curve(mjds, bdts, fluxes, flux_errors, local_backgrounds,
                                                local_background_errors, local_backgrounds_per_pixel,
                                                local_background_errors_per_pixel, flags, fltr, i)
    
    def _extract_annulus_fluxes_from_batch(self, batch: List[str], fltr: str, radius: float, remove_cosmic_rays: bool) -> Tuple[List, List, List, List, List, List, str]:
        """
        Perform aperture photometry with local background subtractions on a single image.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        fltr : str
            The filter.
        radius : float
            The aperture radius.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        
        Returns
        -------
        Tuple[List, List, List, List, List, List, str]
            The source fluxes, source flux errors, local background, local background errors, local background per pixel,
            local background error per pixel, and image quality flag.
        """
        
        batch_fluxes, batch_flux_errors = [], []
        batch_local_backgrounds, batch_local_background_errors = [], []
        batch_local_backgrounds_per_pixel, batch_local_background_errors_per_pixel = [], []
        flags = []
        
        for file in batch:
        
            fluxes, flux_errors = [], []
            local_backgrounds, local_background_errors = [], []
            local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
            
            # if file is the reference image
            if file == self.camera_files[fltr][self.reference_indices[fltr]]:
                flags.append("A")
            # else if file does not have a transform
            elif file not in self.transforms.keys():
                flags.append("B")
            # if file is not the reference image and has a transform
            else:
                flags.append("A")
                transform = self.transforms[file]
            
            data, error = self._get_image_and_error(file, remove_cosmic_rays)
            
            # for each source
            for i in range(len(self.catalogs[fltr])):
                
                catalog_position = (self.catalogs[fltr]["xcentroid"][i], self.catalogs[fltr]["ycentroid"][i])
                
                # try to transform source position
                try:
                    position = matrix_transform(catalog_position, transform)[0]
                # if transform fails, use catalog position
                except:
                    position = catalog_position
                
                flux, flux_error, local_background, local_background_error, local_background_per_pixel, local_background_error_per_pixel = self._compute_annulus_flux(data, error, position, radius)
                fluxes.append(flux)
                flux_errors.append(flux_error)
                local_backgrounds.append(local_background)
                local_background_errors.append(local_background_error)
                local_backgrounds_per_pixel.append(local_background_per_pixel)
                local_background_errors_per_pixel.append(local_background_error_per_pixel)
            
            batch_fluxes.append(fluxes)
            batch_flux_errors.append(flux_errors)
            batch_local_backgrounds.append(local_backgrounds)
            batch_local_background_errors.append(local_background_errors)
            batch_local_backgrounds_per_pixel.append(local_backgrounds_per_pixel)
            batch_local_background_errors_per_pixel.append(local_background_errors_per_pixel)
        
        return batch_fluxes, batch_flux_errors, batch_local_backgrounds, batch_local_background_errors, batch_local_backgrounds_per_pixel, batch_local_background_errors_per_pixel, flags
    
    def _parse_batch_forced_annulus_extraction_results(self, results):
        
        batch_annulus_fluxes, batch_annulus_flux_errors, batch_local_backgrounds, batch_local_background_errors, batch_local_backgrounds_per_pixel, batch_local_background_errors_per_pixel, batch_flags = zip(*results)
        
        annulus_fluxes, annulus_flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
        flags = []
        
        for i in range(len(batch_annulus_fluxes)):
            annulus_fluxes += batch_annulus_fluxes[i]
            annulus_flux_errors += batch_annulus_flux_errors[i]
            local_backgrounds += batch_local_backgrounds[i]
            local_background_errors += batch_local_background_errors[i]
            local_backgrounds_per_pixel += batch_local_backgrounds_per_pixel[i]
            local_background_errors_per_pixel += batch_local_background_errors_per_pixel[i]
            flags += batch_flags[i]
        
        return annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags
    
    def _get_image_and_error(self, file: str, remove_cosmic_rays: bool) -> Tuple[ArrayLike, ArrayLike]:
        """
        Get the image and error for a given file.
        
        Parameters
        ----------
        file : str
            The name of the file.
        
        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            The image and its error.
        """
        
        data = get_data(self.data_directory + file)
        
        if remove_cosmic_rays:
            data = cosmicray_lacosmic(data, gain_apply=False)[0]
        
        error = np.sqrt(data*self.gains[file])  # Poisson noise
        
        return data, error
    
    def _compute_annulus_flux(self, data: ArrayLike, error: ArrayLike, position: ArrayLike, radius: float) -> Tuple[float, float, float, float, float, float]:
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
    
    def _save_annulus_light_curve(self, mjds: ArrayLike, bdts: ArrayLike, fluxes: ArrayLike, flux_errors: ArrayLike,
                                  local_backgrounds: ArrayLike, local_background_errors: ArrayLike,
                                  local_backgrounds_per_pixel: ArrayLike, local_background_errors_per_pixel: ArrayLike,
                                  flags: ArrayLike, fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        mjds : ArrayLike
            The observation MJDs.
        bdts : ArrayLike
            The observation BDTs.
        fluxes : ArrayLike
            The source fluxes.
        flux_errors : ArrayLike
            The source flux errors.
        local_backgrounds : ArrayLike
            The local backgrounds.
        local_background_errors : ArrayLike
            The local background errors.
        local_backgrounds_per_pixel : ArrayLike
            The local backgrounds per pixel.
        local_background_errors_per_pixel : ArrayLike
            The local background errors per pixel.
        flags : ArrayLike
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
        
        fig, axs = plt.subplots(nrows=3, tight_layout=True, figsize=(6.4, 2*4.8), sharex=True, gridspec_kw={"hspace": 0})
        
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
        plt.close(fig)
    
    def _extract_aperture_and_annulus_light_curves(self, batch_size: int, remove_cosmic_rays: bool,
                                                   overwrite: bool) -> None:
        """
        Extract both simple aperture and local-background-subtracted aperture fluxes. This method is more efficient than
        calling _extract_aperture_light_curves() and _extract_annulus_light_curves() separately since it only opens the
        file once.
        
        Parameters
        ----------
        batch_size : int
            The number of images to process in each batch.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        overwrite : bool
            Whether to overwrite existing light curves.
        """
        
        # create output directories if they do not exist
        if not os.path.isdir(self.out_directory + f"aperture_light_curves"):
            os.mkdir(self.out_directory + f"aperture_light_curves")
        if not os.path.isdir(self.out_directory + f"annulus_light_curves"):
            os.mkdir(self.out_directory + f"annulus_light_curves")
        
        if self.verbose:
            print(f"[OPTICAM] Extracting aperture and annulus fluxes ...")
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            if os.path.isfile(self.out_directory + f"aperture_light_curves/{fltr}_source_1.csv") and not overwrite:
                print(f'[OPTICAM] {fltr} light curves already exist. To overwrite, set overwrite to True.')
                continue
            
            try:
                # get aperture radius
                radius = self.scale*self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            except:
                # skip cameras with no sources
                continue
            
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            if self.verbose:
                # perform aperture and annulus photometry in parallel with progress bar
                print(f"[OPTICAM] Processing {fltr} files ...")
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(self._extract_aperture_and_annulus_fluxes_from_batch, fltr=fltr, radius=radius, remove_cosmic_rays=remove_cosmic_rays), batches), total=len(batches)))
                print("[OPTICAM] Done.")
            else:
                # perform aperture and annulus photometry in parallel without progress bar
                with Pool(self.number_of_processors) as pool:
                    results = pool.map(partial(self._extract_aperture_and_annulus_fluxes_from_batch, fltr=fltr, radius=radius, remove_cosmic_rays=remove_cosmic_rays), batches)
            
            # unpack results
            aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = self._parse_batch_forced_aperture_and_annulus_extraction_results(results)
            
            mjds = [self.mjds[file] for file in self.camera_files[fltr]]
            bdts = [self.bdts[file] for file in self.camera_files[fltr]]
            
            if self.verbose:
                print("[OPTICAM] Saving light curves ...")
                for i in tqdm(range(len(self.catalogs[fltr]))):
                    self._save_aperture_light_curve(mjds, bdts, aperture_fluxes, aperture_flux_errors, flags, fltr, i)
                    self._save_annulus_light_curve(mjds, bdts, annulus_fluxes, annulus_flux_errors, local_backgrounds,
                                                local_background_errors, local_backgrounds_per_pixel,
                                                local_background_errors_per_pixel, flags, fltr, i)
                print(f"[OPTICAM] Done.")
            else:
                for i in range(len(self.catalogs[fltr])):
                    self._save_aperture_light_curve(mjds, bdts, aperture_fluxes, aperture_flux_errors, flags, fltr, i)
                    self._save_annulus_light_curve(mjds, bdts, annulus_fluxes, annulus_flux_errors, local_backgrounds,
                                                local_background_errors, local_backgrounds_per_pixel,
                                                local_background_errors_per_pixel, flags, fltr, i)
    
    def _extract_aperture_and_annulus_fluxes_from_batch(self, batch: List[str], fltr: str,radius: float, remove_cosmic_rays: bool) -> Tuple[List, List, List, List, List, List, List, List, str]:
        """
        Extract both simple aperture and local-background-subtracted aperture fluxes from a single image.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        fltr : str
            The filter.
        radius : float
            The aperture radius.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        
        Returns
        -------
        Tuple[List, List, List, List, List, List, List, List, str]
            The fluxes, flux errors, local-background-subtracted fluxes, local-background-subtracted flux errors, local
            backgrounds, local background errors, local backgrounds per pixel, local background errors per pixel, and
            image quality flag.
        """
        
        batch_aperture_fluxes, batch_aperture_flux_errors = [], []
        batch_annulus_fluxes, batch_annulus_flux_errors = [], []
        batch_local_backgrounds, batch_local_background_errors = [], []
        batch_local_backgrounds_per_pixel, batch_local_background_errors_per_pixel = [], []
        flags = []
        
        for file in batch:
            
            aperture_fluxes, aperture_flux_errors = [], []
            annulus_fluxes, annulus_flux_errors = [], []
            local_backgrounds, local_background_errors = [], []
            local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
            
            # if file is the reference image
            if file == self.camera_files[fltr][self.reference_indices[fltr]]:
                flags.append("A")
            # else if file has a transform
            elif file in self.transforms.keys():
                flags.append("A")
                transform = self.transforms[file]
            # else file not aligned
            else:
                flags.append("B")
            
            data = get_data(self.data_directory + file)  # open image
            
            if remove_cosmic_rays:
                data = cosmicray_lacosmic(data, gain_apply=False)[0]
            
            error = np.sqrt(data*self.gains[file])  # Poisson noise
            bkg = self.background(data)  # get background
            clean_data = data - bkg.background  # subtract background
            clean_error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])  # total error
            
            # for each source
            for i in range(len(self.catalogs[fltr])):
                
                # get source catalog position
                catalog_position = (self.catalogs[fltr]["xcentroid"][i], self.catalogs[fltr]["ycentroid"][i])
                
                # get the aligned source position if possible
                try:
                    position = matrix_transform(catalog_position, transform)[0]
                except:
                    position = catalog_position
                
                # get aperture flux
                aperture_flux, aperture_flux_error = self._compute_aperture_flux(clean_data, clean_error, position, radius)
                aperture_fluxes.append(aperture_flux)
                aperture_flux_errors.append(aperture_flux_error)
                
                # get aperture - annulus flux
                annulus_flux, annulus_flux_error, local_background, local_background_error, local_background_per_pixel, local_background_error_per_pixel = self._compute_annulus_flux(data, error, position, radius)
                annulus_fluxes.append(annulus_flux)
                annulus_flux_errors.append(annulus_flux_error)
                local_backgrounds.append(local_background)
                local_background_errors.append(local_background_error)
                local_backgrounds_per_pixel.append(local_background_per_pixel)
                local_background_errors_per_pixel.append(local_background_error_per_pixel)
            
            batch_aperture_fluxes.append(aperture_fluxes)
            batch_aperture_flux_errors.append(aperture_flux_errors)
            batch_annulus_fluxes.append(annulus_fluxes)
            batch_annulus_flux_errors.append(annulus_flux_errors)
            batch_local_backgrounds.append(local_backgrounds)
            batch_local_background_errors.append(local_background_errors)
            batch_local_backgrounds_per_pixel.append(local_backgrounds_per_pixel)
            batch_local_background_errors_per_pixel.append(local_background_errors_per_pixel)
        
        return batch_aperture_fluxes, batch_aperture_flux_errors, batch_annulus_fluxes, batch_annulus_flux_errors, batch_local_backgrounds, batch_local_background_errors, batch_local_backgrounds_per_pixel, batch_local_background_errors_per_pixel, flags
    
    def _parse_batch_forced_aperture_and_annulus_extraction_results(self, results):
        
        batch_aperture_fluxes, batch_aperture_flux_errors, batch_annulus_fluxes, batch_annulus_flux_errors, batch_local_backgrounds, batch_local_background_errors, batch_local_backgrounds_per_pixel, batch_local_background_errors_per_pixel, batch_flags = zip(*results)
        
        aperture_fluxes, aperture_flux_errors = [], []
        annulus_fluxes, annulus_flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
        flags = []
        
        for i in range(len(batch_aperture_fluxes)):
            aperture_fluxes += batch_aperture_fluxes[i]
            aperture_flux_errors += batch_aperture_flux_errors[i]
            annulus_fluxes += batch_annulus_fluxes[i]
            annulus_flux_errors += batch_annulus_flux_errors[i]
            local_backgrounds += batch_local_backgrounds[i]
            local_background_errors += batch_local_background_errors[i]
            local_backgrounds_per_pixel += batch_local_backgrounds_per_pixel[i]
            local_background_errors_per_pixel += batch_local_background_errors_per_pixel[i]
            flags += batch_flags[i]
        
        return aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags




    def photometry(self, phot_type: Literal['both', 'normal', 'optimal'] = 'both',
                   background_method: Literal['global', 'local'] = 'global', tolerance: float = 5.,
                   remove_cosmic_rays: bool = False, overwrite: bool = False) -> None:
        """
        Perform photometry by fitting for the source positions in each image. This method can misidentify sources if the
        field is crowded or the alignments are poor. That said, this method can also yield light curves with higher
        signal-to-noise ratios than forced photometry.
        
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
        remove_cosmic_rays : bool, optional
            Whether to remove cosmic rays from the images before performing photometry, by default False. Removing
            cosmic rays can reduce the number of outliers that appear in the resulting light curves, but doing so can
            significantly increase the processing time.
        overwrite : bool, optional
            Whether to overwrite existing light curves, by default False.
        """
        
        # TODO: change batch size to be set per filter
        n_catalog_files = sum([len(self.camera_files[fltr]) for fltr in list(self.catalogs.keys())])
        batch_size = 1 + int(n_catalog_files/(len(self.catalogs)*self.number_of_processors))
        
        # determine which photometry function to use
        if phot_type == "normal":
            return self._extract_normal_light_curves(background_method, tolerance, batch_size, remove_cosmic_rays,
                                                     overwrite)
        elif phot_type == "optimal":
            return self._extract_optimal_light_curves(background_method, tolerance, batch_size, remove_cosmic_rays,
                                                      overwrite)
        elif phot_type == "both":
            self._extract_normal_and_optimal_light_curves(background_method, tolerance, batch_size, remove_cosmic_rays,
                                                          overwrite)
        else:
            raise ValueError(f"[OPTICAM] Photometry type {phot_type} not recognised.")
    
    def _extract_normal_light_curves(self, background_method: Literal['global', 'local'], tolerance: float,
                                     batch_size: int, remove_cosmic_rays: bool, overwrite: bool) -> None:
        """
        Extract the source fluxes from the images using simple aperture photometry. Unlike the forced photometry methods,
        this method requires fitting for the source positions in each image; as such, this method can be significantly
        slower. Moreover, this method can also misidentify sources if the field is crowded or the alignments are poor.
        
        Parameters
        ----------
        background_method : Literal['global', 'local']
            The method to use for background subtraction. 'global' uses the background attribute to compute the 2D
            background across an entire image, while 'local' uses the local_background attribute to estimate the local
            background around each source.
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the alignments
            are good and the field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields,
            this value can be increased.
        batch_size : int
            The number of images to process in each batch.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        overwrite : bool
            Whether to overwrite existing light curves.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"normal_light_curves"):
            os.mkdir(self.out_directory + f"normal_light_curves")
        
        if self.verbose:
            print(f"[OPTICAM] Extracting normal fluxes ...")
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            #  skip cameras that have already been done
            if os.path.isfile(self.out_directory + f"normal_light_curves/{fltr}_source_1.csv") and not overwrite:
                print(f"[OPTICAM]  Existing {fltr} light curves detected. To overwrite these files, set overwrite to True.")
                continue
            
            # get PSF parameters
            semimajor_sigma = self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            semiminor_sigma = self.aperture_selector(self.catalogs[fltr]["semiminor_sigma"].value)
            
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            if self.verbose:
                print(f"[OPTICAM] Processing {fltr} files ...")
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(self._extract_normal_source_fluxes_from_batch,
                                                        fltr=fltr, semimajor_sigma=semimajor_sigma,
                                                        semiminor_sigma=semiminor_sigma,
                                                        background_method=background_method,
                                                        tolerance=tolerance, remove_cosmic_rays=remove_cosmic_rays),
                                                  batches), total=len(batches)))
                print("[OPTICAM] Done.")
            else:
                with Pool(self.number_of_processors) as pool:
                    results = pool.map(partial(self._extract_normal_source_fluxes_from_batch, fltr=fltr,
                                               semimajor_sigma=semimajor_sigma, semiminor_sigma=semiminor_sigma,
                                               background_method=background_method,
                                               tolerance=tolerance, remove_cosmic_rays=remove_cosmic_rays), batches)
            
            mjds, bdts, fluxes, flux_errors, detections = self._parse_batch_extraction_results(results)
            
            if self.verbose:
                print(f"[OPTICAM] Saving light curves ...")
                for i in tqdm(range(len(self.catalogs[fltr]))):
                    self._save_normal_light_curve(mjds, bdts, fluxes, flux_errors, fltr, i)
                print("[OPTICAM] Done.")
            else:
                for i in range(len(self.catalogs[fltr])):
                    self._save_normal_light_curve(mjds, bdts, fluxes, flux_errors, fltr, i)
            
            self._plot_number_of_detections_per_source(detections, fltr)  # plot number of detections per source
    
    def _extract_normal_source_fluxes_from_batch(self, batch: List[str], fltr: str, semimajor_sigma: float, 
                                                semiminor_sigma: float, background_method: Literal['global', 'local'],
                                                tolerance: float,
                                                remove_cosmic_rays: bool) -> Tuple[float, float, List, List, ArrayLike]:
        """
        Extract the source fluxes from an image using simple aperture photometry. Unlike the forced photometry methods,
        this method requires fitting for the source positions in each image; as such, this method is significantly
        slower. Moreover, this method can also misidentify sources if the field is crowded or the alignments are poor.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        fltr : str
            The filter of the image.
        semimajor_sigma : float
            The semimajor axis of the (presumed 2D Gaussian) PSF.
        semiminor_sigma : float
            The semiminor axis of the (presumed 2D Gaussian) PSF.
        background_method : Literal['global', 'local']
            The method to use for background subtraction. 'global' uses the background attribute to compute the 2D
            background across an entire image, while 'local' uses the local_background attribute to estimate the local
            background around each source.
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the source is
            further than this tolerance, it will be considered a different source. If the alignments are good and the
            field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields, this value can
            be increased.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        
        Returns
        -------
        Tuple[float, float, float, float]
            The observation MJD, BDT, source flux, and source flux error.
        """
        
        batch_fluxes, batch_flux_errors = [], []
        detections = np.zeros(len(self.catalogs[fltr]))
        batch_mjds, batch_bdts = [], []
        
        for file in batch:
            # if file does not have a transform, and it's not the reference image, skip it
            if file not in self.transforms.keys() and file != self.camera_files[fltr][self.reference_indices[fltr]]:
                continue
            
            fluxes, flux_errors = [], []
            
            # load image data
            data = get_data(self.data_directory + file)
            
            # remove cosmic rays if specified
            if remove_cosmic_rays:
                data = cosmicray_lacosmic(data, gain_apply=False)[0]
            
            bkg = self.background(data)
            clean_data = data - bkg.background
            
            if background_method == "global":
                error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])
            else:
                error = np.sqrt(data*self.gains[file])
            
            # find sources in the image
            try:
                segment_map = self.finder(clean_data, self.threshold*bkg.background_rms)
            except:
                continue
            
            # create source catalog
            file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
            file_tbl = file_cat.to_table()
            
            # for each source
            for i in range(len(self.catalogs[fltr])):
                try:
                    # get position of nearest source
                    position = self._get_position_of_nearest_source(file_tbl, i, fltr, file, tolerance)
                except:
                    # if the nearest source exceeds the tolerance, skip it
                    fluxes.append(None)
                    flux_errors.append(None)
                    continue
                
                # count source detection
                detections[i] += 1
                
                # compute source flux
                if background_method == "global":
                    flux, flux_error = self._compute_normal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]["orientation"][i].value)
                else:
                    flux, flux_error = self._compute_normal_flux(data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]["orientation"][i].value, estimate_local_background=True)
                
                fluxes.append(flux)
                flux_errors.append(flux_error)
            
            batch_mjds.append(self.mjds[file])
            batch_bdts.append(self.bdts[file])
            batch_fluxes.append(fluxes)
            batch_flux_errors.append(flux_errors)
        
        return batch_mjds, batch_bdts, batch_fluxes, batch_flux_errors, detections
    
    def _parse_batch_extraction_results(self, results:  List[Tuple[float, float, List, List, ArrayLike]]) -> Tuple[List, List, List, List, ArrayLike]:
        """
        Parse the results of the normal photometry extraction.
        
        Parameters
        ----------
        results : List[Tuple[float, float, List, List, ArrayLike]]
            The results of the normal photometry extraction.
        
        Returns
        -------
        Tuple[List, List, List, List, ArrayLike]
            The observation MJDs, BDTs, source fluxes, source flux errors, and source detections.
        """
        
        batch_mjds, batch_bdts, batch_fluxes, batch_flux_errors, batch_detections = zip(*results)
        
        mjds, bdts = [], []
        fluxes, flux_errors = [], []
        
        for i in range(len(batch_mjds)):
            mjds += batch_mjds[i]
            bdts += batch_bdts[i]
            fluxes += batch_fluxes[i]
            flux_errors += batch_flux_errors[i]
        
        return mjds, bdts, fluxes, flux_errors, np.sum(batch_detections, axis=0)
    
    def _save_normal_light_curve(self, mjds: ArrayLike, bdts: ArrayLike, fluxes: ArrayLike, flux_errors: ArrayLike,
                                 fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        mjds : ArrayLike
            The observation MJDs.
        bdts : ArrayLike
            The observation BDTs.
        fluxes : ArrayLike
            The source fluxes.
        flux_errors : ArrayLike
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
                # if the source was detected
                if fluxes[i][source_index] is not None:
                    csvwriter.writerow([mjds[i], bdts[i], fluxes[i][source_index], flux_errors[i][source_index]])
        
        df = pd.read_csv(self.out_directory + f"normal_light_curves/{fltr}_source_{source_index + 1}.csv")
        
        # reformat MJD to seconds from first observation
        df["time"] = df["MJD"] - self.t_ref
        df["time"] *= 86400
        
        fig, ax = plt.subplots(tight_layout=True, figsize=(6.4, 4.8))
        
        ax.errorbar(df["time"].values, df["flux"].values, yerr=df["flux_error"].values, fmt="k.", ms=2, ecolor="grey",
                    elinewidth=1)
        ax.set_ylabel("Flux [counts]")
        ax.set_title(f"{fltr} Source {source_index + 1}")
        ax.set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
        
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        # save light curve plot to file
        fig.savefig(self.out_directory + f"normal_light_curves/{fltr}_source_{source_index + 1}.png")
        
        plt.close(fig)
    
    def _extract_optimal_light_curves(self, background_method: Literal['global', 'local'], tolerance: float,
                                      batch_size:  int, remove_cosmic_rays: bool, overwrite: bool) -> None:
        """
        Use the optimal photometry method of Naylor 1998, MNRAS, 296, 339 to extract source fluxes from the images.
        Unlike the forced photometry methods, this method requires fitting for the source positions in each image; as
        such, this method can be significantly slower. Moreover, this method can also misidentify sources if the field
        is crowded or the alignments are poor.
        
        Parameters
        ----------
        background_method : Literal['global', 'local']
            The method to use for background subtraction. 'global' uses the background attribute to compute the 2D
            background across an entire image, while 'local' uses the local_background attribute to estimate the local
            background around each source.
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the source is
            further than this tolerance, it will be considered a different source. If the alignments are good and the
            field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields, this value can
            be increased.
        batch_size : int
            The number of images to process in each batch.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        overwrite : bool
            Whether to overwrite existing light curves.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"optimal_light_curves"):
            os.mkdir(self.out_directory + f"optimal_light_curves")
        
        if self.verbose:
            print(f"[OPTICAM] Extracting optimal fluxes ...")
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            #  skip cameras that have already been done
            if os.path.isfile(self.out_directory + f"optimal_light_curves/{fltr}_source_1.csv") and not overwrite:
                print(f"[OPTICAM]  Existing {fltr} light curves detected. To overwrite these files, set overwrite to True.")
                continue
            
            # get PSF parameters
            semimajor_sigma = self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            semiminor_sigma = self.aperture_selector(self.catalogs[fltr]["semiminor_sigma"].value)
            radius = self.scale*self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            if self.verbose:
                print(f"[OPTICAM] Processing {fltr} files ...")
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(self._extract_optimal_source_fluxes_from_batches,
                                                        fltr=fltr, semimajor_sigma=semimajor_sigma,
                                                        semiminor_sigma=semiminor_sigma,
                                                        background_method=background_method, tolerance=tolerance,
                                                        remove_cosmic_rays=remove_cosmic_rays, radius=radius),
                                                  batches), total=len(batches)))
                print("[OPTICAM] Done.")
            else:
                with Pool(self.number_of_processors) as pool:
                    results = pool.map(partial(self._extract_optimal_source_fluxes_from_batches, fltr=fltr,
                                               semimajor_sigma=semimajor_sigma, semiminor_sigma=semiminor_sigma,
                                               tolerance=tolerance, background_method=background_method,
                                               remove_cosmic_rays=remove_cosmic_rays), batches)
            
            # unpack results
            mjds, bdts, fluxes, flux_errors, detections = self._parse_batch_extraction_results(results)
            
            if self.verbose:
                print(f"[OPTICAM] Saving light curves ...")
                for i in tqdm(range(len(self.catalogs[fltr]))):
                    self._save_optimal_light_curve(mjds, bdts, fluxes, flux_errors, fltr, i)
                print("[OPTICAM] Done.")
            else:
                for i in range(len(self.catalogs[fltr])):
                    self._save_optimal_light_curve(mjds, bdts, fluxes, flux_errors, fltr, i)
            
            self._plot_number_of_detections_per_source(detections, fltr)  # plot number of detections per source
    
    def _extract_optimal_source_fluxes_from_batches(self, batch: List[str], fltr: str, semimajor_sigma: float,
                                                 semiminor_sigma: float, background_method: Literal['global', 'local'],
                                                 tolerance: float, remove_cosmic_rays: bool,
                                                 radius: float) -> Tuple[float, float, List, List, ArrayLike]:
        """
        Use the optimal photometry method of Naylor 1998, MNRAS, 296, 339 to extract the source flux from an image.
        Unlike the forced photometry methods, this method requires fitting for the source positions in each image; as
        such, this method can be significantly slower. Moreover, this method can also misidentify sources if the field
        is crowded or the alignments are poor.
        
        Parameters
        ----------
        batches : List[str]
            The list of file names in the batch.
        source : int
            The source number.
        fltr : str
            The filter of the image.
        semimajor_sigma : float
            The semimajor axis of the (presumed 2D Gaussian) PSF.
        semiminor_sigma : float
            The semiminor axis of the (presumed 2D Gaussian) PSF.
        orientation : float
            The orientation of the (presumed 2D Gaussian) PSF.
        background_method : Literal['global', 'local']
            The method to use for background subtraction. 'global' uses the background attribute to compute the 2D
            background across an entire image, while 'local' uses the local_background attribute to estimate the local
            background around each source.
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the source is
            further than this tolerance, it will be considered a different source. If the alignments are good and the
            field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields, this value can
            be increased.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        radius : float
            The aperture radius.
        
        Returns
        -------
        Tuple[float, float, List, List, ArrayLike]
            The observation MJD, BDT, source flux, and source flux error.
        """
        
        batch_fluxes, batch_flux_errors = [], []
        detections = np.zeros(len(self.catalogs[fltr]))
        batch_mjds, batch_bdts = [], []
        
        for file in batch:
            # if file does not have a transform, and it's not the reference image, skip it
            if file not in self.transforms.keys() and file != self.camera_files[fltr][self.reference_indices[fltr]]:
                continue
            
            fluxes, flux_errors = [], []
            
            # load image data
            data = get_data(self.data_directory + file)
            
            # remove cosmic rays if specified
            if remove_cosmic_rays:
                data = cosmicray_lacosmic(data, gain_apply=False)[0]
            
            bkg = self.background(data)
            clean_data = data - bkg.background
            
            if background_method == "global":
                error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])
            else:
                error = np.sqrt(data*self.gains[file])  # Poisson noise
            
            # find sources in the image
            try:
                segment_map = self.finder(clean_data, self.threshold*bkg.background_rms)
            except:
                continue
            
            # create source catalog
            file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
            file_tbl = file_cat.to_table()
            
            # for each source
            for i in range(len(self.catalogs[fltr])):
                try:
                    # get position of nearest source
                    position = self._get_position_of_nearest_source(file_tbl, i, fltr, file, tolerance)
                except:
                    # if the nearest source exceeds the tolerance, skip it
                    fluxes.append(None)
                    flux_errors.append(None)
                    continue
                
                # count source detection
                detections[i] += 1
                
                # compute source flux
                if background_method == "global":
                    flux, flux_error = self._compute_optimal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]["orientation"][i].value, radius)
                else:
                    flux, flux_error = self._compute_optimal_flux(data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]["orientation"][i].value, radius, estimate_local_background=True)
                
                fluxes.append(flux)
                flux_errors.append(flux_error)
            
            batch_mjds.append(self.mjds[file])
            batch_bdts.append(self.bdts[file])
            batch_fluxes.append(fluxes)
            batch_flux_errors.append(flux_errors)
        
        return batch_mjds, batch_bdts, batch_fluxes, batch_flux_errors, detections
    
    def _save_optimal_light_curve(self, mjds: ArrayLike, bdts: ArrayLike, fluxes: ArrayLike, flux_errors: ArrayLike,
                                  fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        mjds : ArrayLike
            The observation MJDs.
        bdts : ArrayLike
            The observation BDTs.
        fluxes : ArrayLike
            The source fluxes.
        flux_errors : ArrayLike
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
                # if the source was detected
                if fluxes[i][source_index] is not None:
                    csvwriter.writerow([mjds[i], bdts[i], fluxes[i][source_index], flux_errors[i][source_index]])
        
        df = pd.read_csv(self.out_directory + f"optimal_light_curves/{fltr}_source_{source_index + 1}.csv")
        
        # reformat MJD to seconds from first observation
        df["time"] = df["MJD"] - self.t_ref
        df["time"] *= 86400
        
        # TODO: add local background axis
        
        fig, ax = plt.subplots(tight_layout=True, figsize=(6.4, 4.8))
        
        ax.errorbar(df["time"].values, df["flux"].values, yerr=df["flux_error"].values, fmt="k.", ms=2, ecolor="grey",
                    elinewidth=1)
        ax.set_ylabel("Flux [counts]")
        ax.set_title(f"{fltr} Source {source_index + 1}")
        ax.set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
        
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        # save light curve plot to file
        fig.savefig(self.out_directory + f"optimal_light_curves/{fltr}_source_{source_index + 1}.png")
        
        plt.close(fig)
    
    def _extract_normal_and_optimal_light_curves(self, background_method: Literal['global', 'local'], tolerance: float,
                                                 batch_size: int, remove_cosmic_rays: bool, overwrite: bool) -> None:
        """
        Extract both normal and optimal source fluxes from the images. This method is more efficient than calling
        _extract_normal_light_curve() and _extract_optimal_light_curve() separately since it only opens the file once.
        
        Parameters
        ----------
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the alignments
            are good and the field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields,
            this value can be increased.
        batch_size : int
            The number of images to process in each batch.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        overwrite : bool
            Whether to overwrite existing light curves.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"normal_light_curves"):
            os.mkdir(self.out_directory + f"normal_light_curves")
        if not os.path.isdir(self.out_directory + f"optimal_light_curves"):
            os.mkdir(self.out_directory + f"optimal_light_curves")
        
        if self.verbose:
            print(f"[OPTICAM] Extracting normal and optimal fluxes ...")
        
        # for each camera
        for fltr in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[fltr]) == 0:
                continue
            
            # skip cameras that have already been done
            if os.path.isfile(self.out_directory + f"normal_light_curves/{fltr}_source_1.csv") and not overwrite:
                print(f"[OPTICAM]  Existing {fltr} light curves detected. To overwrite these files, set overwrite to True.")
                continue
            
            # get PSF parameters
            semimajor_sigma = self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            semiminor_sigma = self.aperture_selector(self.catalogs[fltr]["semiminor_sigma"].value)
            radius = self.scale*self.aperture_selector(self.catalogs[fltr]["semimajor_sigma"].value)
            
            batches = [self.camera_files[fltr][i:i + batch_size] for i in range(0, len(self.camera_files[fltr]), batch_size)]
            
            if self.verbose:
                print(f"[OPTICAM] Processing {fltr} files ...")
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(self._extract_normal_and_optimal_source_fluxes_from_batch,
                                                        fltr=fltr, semimajor_sigma=semimajor_sigma,
                                                        semiminor_sigma=semiminor_sigma,
                                                        background_method=background_method, tolerance=tolerance,
                                                        remove_cosmic_rays=remove_cosmic_rays, radius=radius),
                                                  batches), total=len(batches)))
                print("[OPTICAM] Done.")
            else:
                with Pool(self.number_of_processors) as pool:
                    results = pool.map(partial(self._extract_normal_and_optimal_source_fluxes_from_batch, fltr=fltr,
                                               semimajor_sigma=semimajor_sigma, semiminor_sigma=semiminor_sigma,
                                               background_method=background_method, tolerance=tolerance,
                                               remove_cosmic_rays=remove_cosmic_rays, radius=radius), batches)
            
            # unpack results
            mjds, bdts, normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, detections = self._parse_batch_extraction_results_both(results)
            
            if self.verbose:
                print(f"[OPTICAM] Saving light curves ...")
                for i in tqdm(range(len(self.catalogs[fltr]))):
                    self._save_normal_light_curve(mjds, bdts, normal_fluxes, normal_flux_errors, fltr, i)
                    self._save_optimal_light_curve(mjds, bdts, optimal_fluxes, optimal_flux_errors, fltr, i)
                print("[OPTICAM] Done.")
            else:
                for i in range(len(self.catalogs[fltr])):
                    self._save_normal_light_curve(mjds, bdts, normal_fluxes, normal_flux_errors, fltr, i)
                    self._save_optimal_light_curve(mjds, bdts, optimal_fluxes, optimal_flux_errors, fltr, i)
            
            self._plot_number_of_detections_per_source(detections, fltr)  # plot number of detections per source
    
    def _extract_normal_and_optimal_source_fluxes_from_batch(self, batch: List[str], fltr: str, semimajor_sigma: float,
                                                            semiminor_sigma: float,
                                                            background_method: Literal['global', 'local'],
                                                            tolerance: float,
                                                            remove_cosmic_rays: bool,
                                                            radius: float) -> Tuple[List[float], List[float], List[List[float]], List[List[float]], List[List[float]], List[List[float]], ArrayLike]:
        """
        Extract both normal and optimal source fluxes from an image. This method is more efficient than calling
        _extract_normal_light_curve() and _extract_optimal_light_curve() separately since it only opens the file once.
        
        Parameters
        ----------
        batch : List[str]
            The list of file names in the batch.
        source : int
            The source number.
        fltr : str
            The filter of the image.
        semimajor_sigma : float
            The semimajor axis of the (presumed 2D Gaussian) PSF.
        semiminor_sigma : float
            The semiminor axis of the (presumed 2D Gaussian) PSF.
        orientation : float
            The orientation of the (presumed 2D Gaussian) PSF.
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the source is
            further than this tolerance, it will be considered a different source. If the alignments are good and the
            field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields, this value can
            be increased.
        remove_cosmic_rays : bool
            Whether to remove cosmic rays from the images before performing photometry. Removing cosmic rays can reduce
            the number of outliers that appear in the resulting light curves, but doing so can significantly increase the
            processing time.
        radius : float
            The aperture radius.
        
        Returns
        -------
        Tuple[float, float, List, List, List, List, ArrayLike]
            The observation MJD, BDT, normal source flux, normal source flux error, optimal source flux, optimal source
            flux error, and detections.
        """
        
        batch_mjds, batch_bdts = [], []
        batch_normal_fluxes, batch_normal_flux_errors = [], []
        batch_optimal_fluxes, batch_optimal_flux_errors = [], []
        detections = np.zeros(len(self.catalogs[fltr]))
        
        for file in batch:
            # if file does not have a transform, and it's not the reference image, skip it
            if file not in self.transforms.keys() and file != self.camera_files[fltr][self.reference_indices[fltr]]:
                continue
            
            normal_fluxes, normal_flux_errors = [], []
            optimal_fluxes, optimal_flux_errors = [], []
            
            # load image data
            data = get_data(self.data_directory + file)
            
            # remove cosmic rays if specified
            if remove_cosmic_rays:
                data = cosmicray_lacosmic(data, gain_apply=False)[0]
            
            bkg = self.background(data)
            clean_data = data - bkg.background
            
            if background_method == "global":
                error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])
            else:
                error = np.sqrt(data*self.gains[file])  # Poisson noise
            
            # find sources in the image
            try:
                segment_map = self.finder(clean_data, self.threshold*bkg.background_rms)
            except:
                continue
            
            # create source catalog
            file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
            file_tbl = file_cat.to_table()
            
            # for each source
            for i in range(len(self.catalogs[fltr])):
                try:
                    # get position of nearest source
                    position = self._get_position_of_nearest_source(file_tbl, i, fltr, file, tolerance)
                except:
                    # if the nearest source exceeds the tolerance, skip it
                    normal_fluxes.append(None)
                    normal_flux_errors.append(None)
                    optimal_fluxes.append(None)
                    optimal_flux_errors.append(None)
                    continue
                
                # count source detection
                detections[i] += 1
                
                # compute source flux
                if background_method == 'global':
                    normal_flux, normal_flux_error = self._compute_normal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]["orientation"][i].value)
                else:
                    normal_flux, normal_flux_error = self._compute_normal_flux(data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]["orientation"][i].value, estimate_local_background=True)
                normal_fluxes.append(normal_flux)
                normal_flux_errors.append(normal_flux_error)            
                
                if background_method == 'global':
                    optimal_flux, optimal_flux_error = self._compute_optimal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]["orientation"][i].value, radius)
                else:
                    optimal_flux, optimal_flux_error = self._compute_optimal_flux(data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[fltr]["orientation"][i].value, radius, estimate_local_background=True)
                optimal_fluxes.append(optimal_flux)
                optimal_flux_errors.append(optimal_flux_error)
            
            batch_mjds.append(self.mjds[file])
            batch_bdts.append(self.bdts[file])
            batch_normal_fluxes.append(normal_fluxes)
            batch_normal_flux_errors.append(normal_flux_errors)
            batch_optimal_fluxes.append(optimal_fluxes)
            batch_optimal_flux_errors.append(optimal_flux_errors)
        
        return batch_mjds, batch_bdts, batch_normal_fluxes, batch_normal_flux_errors, batch_optimal_fluxes, batch_optimal_flux_errors, detections
    
    def _compute_normal_flux(self, data: ArrayLike, error: ArrayLike, position: ArrayLike, semimajor_sigma: float,
                             semiminor_sigma: float, orientation: float,
                             estimate_local_background: bool = False) -> Tuple[float, float]:
        """
        Compute the flux at a given position using simple aperture photometry.
        
        Parameters
        ----------
        clean_data : ArrayLike
            The image.
        error : ArrayLike
            The total error in the image.
        position : ArrayLike
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
    
    def _compute_optimal_flux(self, data: NDArray, error: NDArray, position: ArrayLike, semimajor_sigma: float,
                                semiminor_sigma: float, orientation: float, radius: float,
                                estimate_local_background: bool = False) -> Tuple[float, float]:
        """
        Compute the flux at a given position using the optimal photometry method of Naylor 1998, MNRAS, 296, 339.
        
        Parameters
        ----------
        data : NDArray
            The image.
        error : NDArray
            The total error in the image.
        position : ArrayLike
            The aperture position.
        semimajor_sigma : float
            The semimajor axis of the (presumed 2D Gaussian) PSF.
        semiminor_sigma : float
            The semiminor axis of the (presumed 2D Gaussian) PSF.
        orientation : float
            The orientation of the (presumed 2D Gaussian) PSF.
        radius : float
            The aperture radius.
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
        
        return np.sum(clean_data*weights), np.sqrt(np.sum((error*weights)**2))
    
    def _get_position_of_nearest_source(self, file_tbl: QTable, source_index: int, fltr: str, file: str,
                                        tolerance: float) -> ArrayLike:
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
        ArrayLike
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
    
    def _plot_number_of_detections_per_source(self, detections: ArrayLike, fltr: str) -> None:
        """
        Plot the number of detections per source.
        
        Parameters
        ----------
        detections : ArrayLike
            The number of detections per source.
        fltr : str
            The filter used to observe the sources.
        phot_type : Literal['normal', 'optimal']
            The type of photometry used to extract the light curves.
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
    
    def _parse_batch_extraction_results_both(self, results):
        
        batch_mjds, batch_bdts, batch_normal_fluxes, batch_normal_flux_errors, batch_optimal_fluxes, batch_optimal_flux_errors, batch_detections = zip(*results)
        
        mjds, bdts = [], []
        normal_fluxes, normal_flux_errors = [], []
        optimal_fluxes, optimal_flux_errors = [], []
        
        for i in range(len(batch_mjds)):
            mjds += batch_mjds[i]
            bdts += batch_bdts[i]
            normal_fluxes += batch_normal_fluxes[i]
            normal_flux_errors += batch_normal_flux_errors[i]
            optimal_fluxes += batch_optimal_fluxes[i]
            optimal_flux_errors += batch_optimal_flux_errors[i]
        
        return mjds, bdts, normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, np.sum(batch_detections, axis=0)






