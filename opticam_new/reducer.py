from tqdm import tqdm
from astropy.table import QTable
import json
import astroalign
import numpy as np
from astropy.time import Time
import os
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.visualization.mpl_normalize import simple_norm
from photutils.segmentation import SourceCatalog
from photutils.aperture import ApertureStats, aperture_photometry, CircularAperture, CircularAnnulus, EllipticalAperture
from photutils.utils import calc_total_error
from skimage.transform import SimilarityTransform
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image
from typing import List, Dict, Literal, Callable, Tuple
from numpy.typing import ArrayLike
import pandas as pd
import csv
import warnings

from opticam_new.helpers import get_data, log_binnings, log_filters, default_aperture_selector
from opticam_new.background import Background
from opticam_new.finder import Finder


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
        finder: Callable = None,
        aperture_selector: Callable = None,
        scale: float = 5,
        r_in_scale: float = 1,
        r_out_scale: float = 2,
        local_background_method: Literal["mean", "median"] = "mean",
        local_background_sigma_clip: SigmaClip = SigmaClip(sigma=3, maxiters=10),
        number_of_processors: int = int(cpu_count()/2),
        show_plots: bool = True,
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
        finder: Callable, optional
            The source finder, by default None. If None, the default source finder is used.
        aperture_selector: Callable, optional
            The aperture selector, by default None. If None, the default aperture selector is used.
        scale: float, optional
            The aperture scale factor, by default 5. The aperture scale factor scales the aperture size returned by
            aperture_selector for forced photometry.
        r_in_scale: float, optional
            The inner radius scale factor for the annulus used for local background estimation, by default 1 (equal to 
            the aperture radius).
        r_out_scale: float, optional
            The outer radius scale factor for the annulus used for local background estimation, by default 2 (equal to 
            twice the aperture radius).
        local_background_method: Literal["mean", "median"], optional
            The method for estimating the local background, by default "mean". The local background is estimated as the 
            mean or median of the pixel values in the annulus, and the associated error is estimated as the standard
            deviation or the median absolute deviation of the pixel values, respectively.
        local_background_sigma_clip: SigmaClip, optional
            The sigma clipper for the local background estimation, by default
            astropy.stats.SigmaClip(sigma=3, maxiters=10).
        number_of_processors: int, optional
            The number of processors to use for parallel processing, by default half the number of available processors.
            Note that there is some overhead incurred when using multiple processors, so there can be diminishing
            returns when using more processors.
        show_plots: bool, optional
            Whether to show plots as they're created, by default True. Whether True or False, plots are always saved
            to out_directory.
        
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
        
        self.data_directory = data_directory
        if not self.data_directory[-1].endswith("/"):
            self.data_directory += "/"
        
        self.out_directory = out_directory
        if not self.out_directory.endswith("/"):
            self.out_directory += "/"
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory):
            print(f"[OPTICAM] {self.out_directory} not found, attempting to create ...")
            # create output directory if it does not exist
            directories = self.out_directory.split("/")[1:-1]  # remove leading and trailing slashes
            for i in range(len(directories)):
                if not os.path.isdir("/" + "/".join(directories[:i + 1])):
                    try:
                        os.mkdir("/" + "/".join(directories[:i + 1]))
                    except:
                        raise FileNotFoundError(f"[OPTICAM] Could not create directory {directories[:i + 1]}")
            print(f"[OPTICAM] {self.out_directory} created.")
        
        # create subdirectories
        if not os.path.isdir(self.out_directory + "cat"):
            os.mkdir(self.out_directory + "cat")
        if not os.path.isdir(self.out_directory + "diag"):
            os.mkdir(self.out_directory + "diag")
        if not os.path.isdir(self.out_directory + "misc"):
            os.mkdir(self.out_directory + "misc")
        
        # set parameters
        self.aperture_selector = default_aperture_selector if aperture_selector is None else aperture_selector
        self.scale = scale
        self.r_in_scale = r_in_scale
        self.r_out_scale = r_out_scale
        self.threshold = threshold
        self.local_background_method = local_background_method
        self.local_background_sigma_clip = local_background_sigma_clip
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        assert callable(self.aperture_selector), "[OPTICAM] Aperture selector must be callable."
        assert isinstance(self.scale, (int, float)), "[OPTICAM] Aperture scale must be an integer or float."
        assert isinstance(self.r_in_scale, (int, float)), "[OPTICAM] Annulus r_in scale must be an integer or float."
        assert isinstance(self.r_out_scale, (int, float)), "[OPTICAM] Annulus r_out scale must be an integer or float."
        assert isinstance(self.threshold, (int, float)), "[OPTICAM] Threshold must be an integer or float."
        assert local_background_method in ["mean", "median"], "[OPTICAM] Local background method must be 'mean' or 'median'."
        # validate local background sigma clip
        assert isinstance(self.number_of_processors, int), "[OPTICAM] Number of processors must be an integer."
        assert isinstance(self.show_plots, bool), "[OPTICAM] Show plots must be a boolean."
        
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
            "annulus r_in scale": r_in_scale,
            "annulus r_out scale": r_out_scale,
            "threshold": threshold,
            "local background method": local_background_method,
        }
        try:
            for key, value in self.local_background_sigma_clip.__dict__.items():
                if not key.startswith("_"):
                    param_dict["local background SigmaClip " + str(key)] = value
        except:
            pass
        param_dict.update({"number of files": len(self.file_names)})
        param_dict.update({f"number of {fltr} files": len(self.camera_files[f"{fltr}"]) for fltr in list(self.camera_files.keys())})
        with open(self.out_directory + "misc/reducer_input.json", "w") as file:
            json.dump(param_dict, file, indent=4)
        
        # define background calculator and write input parameters to file
        if background is None:
            if self.binning not in ["8x8", "4x4", "3x3", "2x2", "1x1"]:
                raise ValueError(f"[OPTICAM] Binning {self.binning} is not (yet) supported! Supported binning values are \"8x8\" \"4x4\", \"3x3\", \"2x2\", and \"1x1\".")
            box_size = 8 if self.binning == "8x8" else 16 if self.binning == "4x4" else 22 if self.binning == "3x3" else 32 if self.binning == "2x2" else 64  # set box_size based on binning
            self.background = Background(box_size=box_size)
        else:
            self.background = background
        try:
            with open(self.out_directory + "misc/background_input.json", "w") as file:
                json.dump(self.background.get_input_dict(), file, indent=4)
        except:
            warnings.warn("[OPTICAM] Could not write background input parameters to file. It's a good idea to add a get_input_dict() method to your background estimator for reproducability (see the background tutorial).")
        
        # TODO: automate npixels/border_width/box_size based on binning more generically (e.g., by scaling a value for 1x1)
        # define source finder and write input parameters to file
        if finder is None:
            if self.binning not in ["8x8", "4x4", "3x3", "2x2", "1x1"]:
                raise ValueError(f"[OPTICAM] Binning {self.binning} is not (yet) supported! Supported binning values are \"8x8\" \"4x4\", \"3x3\", \"2x2\", and \"1x1\".")
            npixels = 12 if self.binning == "8x8" else 25 if self.binning == "4x4" else 33 if self.binning == "3x3" else 50 if self.binning == "2x2" else 100
            border_width = 8 if self.binning == "8x8" else 16 if self.binning == "4x4" else 22 if self.binning == "3x3" else 32 if self.binning == "2x2" else 64
            self.finder = Finder(npixels=npixels, border_width=border_width)
        else:
            self.finder = finder
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
            print("[OPTICAM] Read transforms from file.")
        except:
            pass
        
        # try to load catalogs from file
        for fltr in list(self.camera_files.keys()):
            try:
                self.catalogs.update({f"{fltr}": QTable.read(self.out_directory + f"cat/{fltr}_catalog.ecsv", format="ascii.ecsv")})
                print(f"[OPTICAM] Read {fltr} catalog from file.")
                continue
            except:
                pass
    
    def _scan_data_directory(self) -> None:
        """
        Scan the data directory for files and extract the MJD, filter, binning, and gain from the file headers.
        
        Raises
        ------
        ValueError
            If more than 3 filters are found.
        ValueError
            If the binning is not consistent.
        """
        
        print("[OPTICAM] Scanning files ...")
        
        self.times = {}  # file name : file MJD
        self.gains = {}  # file name : file gain
        self.camera_files = {}  # filter : [files]
        
        with Pool(self.number_of_processors) as pool:
            results = list(tqdm(pool.imap(self._scan_file, self.file_names), total=len(self.file_names)))
        
        # unpack results
        file_mjds, file_filters, file_binnings, file_gains = zip(*results)
        
        filters = sorted(list(set(file_filters)))  # get unique filters
        if len(filters) > 3:
            log_filters(self.data_directory, self.out_directory)
            raise ValueError("[OPTICAM] More than 3 filters found. Image filters have been logged to {self.out_directory}misc/filters.json.")
        else:
            with open(self.out_directory + "misc/filters.txt", "w") as file:
                for fltr in filters:
                    file.write(f"{fltr}-band\n")
        
        binning = list(set(file_binnings))
        # if binning is not consistent, raise error
        if len(binning) > 1:
            log_binnings(self.data_directory, self.out_directory)
            raise ValueError(f"[OPTICAM] All images must have the same binning. Image binnings have been logged to {self.out_directory}diag/binnings.json.")
        else:
            self.binning = binning[0]
        
        # for each filter
        for fltr in filters:
            self.camera_files.update({f"{fltr}-band": []})  # initialise list of files for each filter in a dictionary
            
            # for each file
            for i in range(len(self.file_names)):
                # if the file filter matches the current filter
                if file_filters[i] == fltr:
                    self.camera_files[f"{fltr}-band"].append(self.file_names[i])  # add file name to dict list
                
                # if the file gain is not already in the dict
                if self.file_names[i] not in self.gains.keys():
                    self.gains.update({self.file_names[i]: file_gains[i]})  # add file gain to dict
                
                # if the file time is not already in the dict
                if self.file_names[i] not in self.times.keys():
                    self.times.update({self.file_names[i]: file_mjds[i]})  # add file time to dict
        
        # sort files by time
        for key in list(self.camera_files.keys()):
            self.camera_files[key].sort(key=lambda x: self.times[x])
        
        self.t_ref = min(list(self.times.values()))  # get reference time
        with open(self.out_directory + "misc/earliest_observation_time.txt", "w") as file:
            file.write(str(self.t_ref))
        
        print("[OPTICAM] Done.")
        print(f"[OPTICAM] Binning: {binning}")
    
    def _scan_file(self, file: str) -> Tuple[float, str, str, float]:
        """
        Get the MJD, filter, binning, and gain from the file header.

        Parameters
        ----------
        file : str
            The name of the file.

        Returns
        -------
        Tuple[float, str, str, float]
            The MJD, filter, binning, and gain.

        Raises
        ------
        KeyError
            If the file header does not contain the required keys.
        """
        
        with fits.open(self.data_directory + file) as hdul:
            binning = hdul[0].header["BINNING"]
            gain = hdul[0].header["GAIN"]
            
            # parse file time
            if "GPSTIME" in hdul[0].header.keys():
                gpstime = hdul[0].header["GPSTIME"]
                split_gpstime = gpstime.split(" ")
                date = split_gpstime[0]  # get date
                time = split_gpstime[1].split(".")[0]  # get time (ignoring decimal seconds)
                mjd = Time(date + "T" + time, format="fits").mjd
            elif "UT" in hdul[0].header.keys():
                mjd = Time(hdul[0].header["UT"].replace(" ", "T"), format="fits").mjd
            else:
                raise KeyError(f"[OPTICAM] Could not find GPSTIME or UT key in {file} header.")
            
            # separate files by filter
            fltr = hdul[0].header["FILTER"]
        
        return mjd, fltr, binning, gain




    def initialise_catalogs(self, batch_size: int = None) -> None:
        """
        Initialise the source catalogs for each camera. Some aspects of this method are parallelised for speed.
        
        Parameters
        ----------
        batch_size : int, optional
            The number of images to process in each batch, by default the batch size is based on the number of images, 
            the number of unique filters, and the number of processors. For the best performance, the number of batches
            should not be less than the number of processors.
        """
        
        # automatically determine a suitable batch size
        if batch_size is None:
            batch_size = min(100, 1 + int(len(self.file_names)/(len(self.camera_files)*self.number_of_processors)))
        
        print("[OPTICAM] Initialising catalogs ...")
        
        cat_fig, cat_ax = plt.subplots(ncols=3, tight_layout=True, figsize=(15, 5))
        bkg_fig, bkg_axs = plt.subplots(ncols=3, tight_layout=True, figsize=(15, 5))
        
        background_median = {}
        background_rms = {}
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # if no images found for camera, skip
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            cat_ax[list(self.camera_files.keys()).index(fltr)].set_title(f"{fltr}")
            cat_ax[list(self.camera_files.keys()).index(fltr)].set_xlabel("X")
            cat_ax[list(self.camera_files.keys()).index(fltr)].set_ylabel("Y")
            
            bkg_axs[list(self.camera_files.keys()).index(fltr)].set_title(f"{fltr}")
            
            reference_image = get_data(self.data_directory + self.camera_files[f"{fltr}"][0])  # get reference image
            
            # split files into batches for reduced memory usage
            batches = [self.camera_files[f"{fltr}"][i:i + batch_size] for i in range(0, len(self.camera_files[f"{fltr}"]), batch_size)]
            
            # align and stack images
            with Pool(self.number_of_processors) as pool:
                print(f"[OPTICAM] Aligning and stacking {fltr} images in batches ...")
                results = list(tqdm(pool.imap(partial(self._align_and_stack_image_batch, reference_image=reference_image), batches), total=len(batches)))
            
            # parse batch results
            stacked_image, background_median[f"{fltr}"], background_rms[f"{fltr}"] = self._parse_batch_alignment_and_stacking_results(results, reference_image.copy())
            
            # remove background from stacked images
            stacked_bkg = self.background(stacked_image)
            stacked_image -= stacked_bkg.background
            stacked_image_plot = np.clip(stacked_image, 0, None)  # clip negative values to zero for better visualisation
            
            # plot background mesh
            bkg_axs[list(self.camera_files.keys()).index(fltr)].imshow(stacked_image_plot, origin="lower",
                                                                       cmap="Greys_r", interpolation="nearest",
                                                                       norm=simple_norm(stacked_image_plot, stretch="log"))
            stacked_bkg.plot_meshes(ax=bkg_axs[list(self.camera_files.keys()).index(fltr)], outlines=True, marker='.',
                                    color='cyan', alpha=0.3)
            
            try:
                # identify sources in stacked image
                segment_map = self.finder(stacked_image, self.threshold*stacked_bkg.background_rms)
            except:
                print(f"[OPTICAM] No sources found in {fltr}.")
                continue
            
            # create catalog of sources in stacked image and write to file
            self.catalogs.update({f"{fltr}": SourceCatalog(stacked_image, segment_map,
                                                           background=stacked_bkg.background).to_table()})
            self.catalogs[f"{fltr}"].write(self.out_directory + f"cat/{fltr}_catalog.ecsv", format="ascii.ecsv",
                                            overwrite=True)
            
            # plot stacked image
            cat_ax[list(self.camera_files.keys()).index(fltr)].imshow(stacked_image_plot, origin="lower",
                                                                      cmap="Greys_r", interpolation="nearest",
                                                                      norm=simple_norm(stacked_image_plot, stretch="log"))
            
            # get aperture radius
            radius = self.scale*self.aperture_selector(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            
            # circle and label sources in stacked image            
            for i in range(len(self.catalogs[f"{fltr}"])):
                cat_ax[list(self.camera_files.keys()).index(fltr)].add_patch(Circle(xy=(self.catalogs[f"{fltr}"]["xcentroid"][i],
                                                    self.catalogs[f"{fltr}"]["ycentroid"][i]),
                                                    radius=radius,
                                                    edgecolor=self.colours[i % len(self.colours)],
                                                    facecolor="none",
                                                    lw=1))
                cat_ax[list(self.camera_files.keys()).index(fltr)].add_patch(Circle(xy=(self.catalogs[f"{fltr}"]["xcentroid"][i],
                                                    self.catalogs[f"{fltr}"]["ycentroid"][i]),
                                                    radius=self.r_in_scale*radius,
                                                    edgecolor=self.colours[i % len(self.colours)],
                                                    facecolor="none",
                                                    lw=1, ls=":"))
                cat_ax[list(self.camera_files.keys()).index(fltr)].add_patch(Circle(xy=(self.catalogs[f"{fltr}"]["xcentroid"][i],
                                                    self.catalogs[f"{fltr}"]["ycentroid"][i]),
                                                    radius=self.r_out_scale*radius,
                                                    edgecolor=self.colours[i % len(self.colours)],
                                                    facecolor="none",
                                                    lw=1, ls=":"))
                cat_ax[list(self.camera_files.keys()).index(fltr)].text(self.catalogs[f"{fltr}"]["xcentroid"][i] + 1.05*radius,
                                    self.catalogs[f"{fltr}"]["ycentroid"][i] + 1.05*radius, i + 1,
                                    color=self.colours[i % len(self.colours)])
        
        # save figures
        cat_fig.savefig(self.out_directory + "cat/catalogs.png")
        bkg_fig.savefig(self.out_directory + "diag/background_meshes.png")
        
        self._plot_time_between_files()  # plot time between observations
        self._plot_backgrounds(background_median, background_rms)  # plot backgrounds
        
        # save transforms to file
        with open(self.out_directory + "cat/transforms.json", "w") as file:
            json.dump(self.transforms, file, indent=4)
        
        # write unaligned files to file
        if len(self.unaligned_files) > 0:
            with open(self.out_directory + "diag/unaligned_files.txt", "w") as unaligned_file:
                for file in self.unaligned_files:
                    unaligned_file.write(file + "\n")
        
        # either show or close plots
        if self.show_plots:
            plt.show(cat_fig)
            plt.show(bkg_fig)
        else:
            plt.close(cat_fig)
            plt.close(bkg_fig)
    
    def _align_and_stack_image_batch(self, batch: List[str], reference_image: ArrayLike) -> Tuple[Dict[str, List], List, ArrayLike, List, List]:
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
            
            # try to align image
            try:
                transform = astroalign.find_transform(reference_image, data)[0].params.tolist()  # align image w.r.t reference image
                transforms.update({file: transform})  # store transform in dictionary
                stacked_image += astroalign.apply_transform(SimilarityTransform(transform), data, reference_image)[0]  # align and stack image
            except:
                unaligned_files.append(file)  # store unaligned file in list
            
            # get background
            background = self.background(data)
            background_median.append(background.background_median)
            background_rms.append(background.background_rms_median)
        
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
        
        print(f"[OPTICAM] Done. {len(unaligned_files)} image(s) could not be aligned.")
        
        return stacked_image, background_median, background_rms
        
    def _plot_time_between_files(self) -> None:
        """
        Plot the times between each file for each camera.
        """
        
        fig, axs = plt.subplots(nrows=2, ncols=3, tight_layout=True, figsize=(2*6.4, 2*4.8))
        
        for fltr in list(self.camera_files.keys()):
            times = np.array([self.times[file] for file in self.camera_files[f"{fltr}"]])
            times -= times.min()
            times *= 86400  # convert to seconds from first observation
            dt = np.diff(times)  # get time between files
            file_numbers = np.arange(2, len(times) + 1, 1)  # start from 2 because we are plotting the time between files
            
            bin_edges = np.arange(0, int(dt.max()) + 2, 1)  # define bins with width 1 s
            
            axs[0, list(self.camera_files.keys()).index(fltr)].set_title(f"{fltr}")
            
            axs[0, list(self.camera_files.keys()).index(fltr)].plot(file_numbers, dt, "k-", lw=1)
            
            axs[1, list(self.camera_files.keys()).index(fltr)].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
            axs[1, list(self.camera_files.keys()).index(fltr)].set_yscale("log")
        
        axs[0, 0].set_ylabel("Time between files [s]")
        axs[0, 1].set_xlabel("File number")
        axs[1, 0].set_ylabel("N")
        axs[1, 1].set_xlabel("Time between files [s]")
        
        for ax in axs.flatten():
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
        fig.savefig(self.out_directory + "diag/header_times.png")
    
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
        
        fig, axs = plt.subplots(nrows=2, ncols=3, tight_layout=True, sharex=True, figsize=(2*6.4, 2*4.8))
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            times = np.array([self.times[file] for file in self.camera_files[f"{fltr}"]])  # get times
            plot_times = (times - self.t_ref)*86400  # convert to seconds from first observation
            
            # plot background
            axs[0, list(self.camera_files.keys()).index(fltr)].set_title(f"{fltr}")
            axs[0, list(self.camera_files.keys()).index(fltr)].plot(plot_times, background_rms[f"{fltr}"], "k.", ms=2)
            axs[1, list(self.camera_files.keys()).index(fltr)].plot(plot_times, background_median[f"{fltr}"], "k.", ms=2)
            
            # write background to file
            with open(self.out_directory + f"diag/{fltr}_background.txt", "w") as file:
                file.write("# MJD RMS median\n")
                for i in range(len(self.camera_files[f"{fltr}"])):
                    file.write(f"{times[i]} {background_rms[f"{fltr}"][i]} {background_median[f"{fltr}"][i]}\n")
        
        axs[0, 0].set_ylabel("Median background RMS")
        axs[1, 0].set_ylabel("Median background")
        axs[1, 1].set_xlabel(f"Time from MJD {times.min():.4f} [s]")
        
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




    def create_gifs(self, keep_frames: bool = True) -> None:
        """
        Create alignment gifs for each camera. Some aspects of this method are parallelised for speed. The frames are 
        saved in out_directory/diag/*-band_gif_frames and the GIFs are saved in out_directory/cat.
        
        Parameters
        ----------
        keep_frames : bool, optional
            Whether to save the GIF frames in out_directory/diag, by default True. If False, the frames will be deleted
            after the GIF is saved.
        """
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            else:
                # create gif frames directory if it does not exist
                if not os.path.isdir(self.out_directory + f"diag/{fltr}_gif_frames"):
                    os.mkdir(self.out_directory + f"diag/{fltr}_gif_frames")
            
            # create gif frames
            with Pool(self.number_of_processors) as pool:
                print(f"[OPTICAM] Creating {fltr} GIF frames ...")
                results = list(tqdm(pool.imap(partial(self._create_gif_frame, fltr=fltr), self.camera_files[f"{fltr}"]), total=len(self.camera_files[f"{fltr}"])))
                print(f"[OPTICAM] Done.")
            
            # save GIF
            self._compile_gif(fltr, keep_frames)
    
    def _create_gif_frame(self, file: str, fltr: str) -> None:
        """
        Create a gif frame from the image and save it to the out_directory.

        Parameters
        ----------
        image : np.array
            The image to be saved as a gif frame.
        file_name : str
            The name of the image file.
        camera : int
            The camera number (1, 2, or 3).
        """
        
        data = get_data(self.data_directory + file)
        bkg = self.background(data)
        clean_data = data - bkg.background
        
        fig, ax = plt.subplots(num=999, clear=True, tight_layout=True)  # set figure number to 999 to avoid conflict with other figures
        
        ax.imshow(clean_data, origin="lower", cmap="Greys_r", interpolation="nearest",
                  norm=simple_norm(clean_data, stretch="log"))
        
        # for each source
        for i in range(len(self.catalogs[f"{fltr}"])):
            
            source_position = (self.catalogs[f"{fltr}"]["xcentroid"][i], self.catalogs[f"{fltr}"]["ycentroid"][i])
            
            try:
                aperture_position = astroalign.matrix_transform(source_position, self.transforms[file])[0]
                title = f"{file} (aligned)"
                colour = "black"
            except:
                aperture_position = source_position
                title = f"{file} (unaligned)"
                colour = "red"
            
            radius = self.scale*self.aperture_selector(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            
            ax.add_patch(Circle(xy=(aperture_position), radius=radius,
                                    edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1))
            ax.add_patch(Circle(xy=(aperture_position), radius=self.r_in_scale*radius,
                                edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1, ls=":"))
            ax.add_patch(Circle(xy=(aperture_position), radius=self.r_out_scale*radius,
                                edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1, ls=":"))
            ax.text(aperture_position[0] + 1.05*radius, aperture_position[1] + 1.05*radius, i + 1,
                        color=self.colours[i % len(self.colours)])
        
        ax.set_title(title, color=colour)
        
        fig.savefig(self.out_directory + f"diag/{fltr}_gif_frames/{file.split(".")[0]}.png")
        
        fig.clear()
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
        
        print(f"[OPTICAM] Saving GIF (this can take some time) ...")
        
        # load frames
        frames = []
        for file in self.camera_files[f"{fltr}"]:
            try:
                frames.append(Image.open(self.out_directory + f"diag/{fltr}_gif_frames/{file.split(".")[0]}.png"))
            except:
                pass
        
        # save gif
        frames[0].save(self.out_directory + f"cat/{fltr}_images.gif", format="GIF", append_images=frames[1:], 
                       save_all=True, duration=200, loop=0)
        del frames  # delete frames after gif is saved to clear memory
        
        # delete frames after gif is saved
        if not keep_frames:
            for file in os.listdir(self.out_directory + f"diag/{fltr}_gif_frames"):
                os.remove(self.out_directory + f"diag/{fltr}_gif_frames/{file}")
        
        print("[OPTICAM] Done.")
    
    def _load_gif_frame(self, fltr: str, file_name: str) -> Image:
        """
        Load a gif frame from out_directory.

        Parameters
        ----------
        fltr : str
            The filter.
        file_name : str
            The name of the file whose frame is to be loaded.

        Returns
        -------
        Image
            The gif frame.
        """
        
        return Image.open(self.out_directory + f"diag/{fltr}_gif_frames/{file_name.split(".")[0]}.png")




    def forced_photometry(self, phot_type: Literal["aperture", "annulus", "both"] = "both") -> None:
        """
        Perform forced photometry on the images in out_directory to extract source fluxes.
        
        Parameters
        ----------
        phot_type : Literal["aperture", "annulus", "both"], optional
            The type of photometry to perform, by default "both". If "aperture", only aperture photometry is performed.
            If "annulus", only annulus photometry is performed. If "both", both aperture and annulus photometry are
            performed simultaneously (this is more efficient that performing both separately since it only opens the
            file once).
        
        Raises
        ------
        ValueError
            If phot_type is not recognised.
        """
        
        # determine which photometry function to use
        if phot_type == "aperture":
            self._extract_aperture_light_curves()
        elif phot_type == "annulus":
            self._extract_annulus_light_curves()
        elif phot_type == "both":
            self._extract_aperture_and_annulus_light_curves()
        else:
            raise ValueError(f"[OPTICAM] Photometry type {phot_type} not recognised.")
    
    def _extract_aperture_light_curves(self) -> None:
        """
        Perform forced simple aperture photometry on all the images in out_directory.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"aperture_light_curves"):
            os.mkdir(self.out_directory + f"aperture_light_curves")
        
        print(f"[OPTICAM] Extracting aperture fluxes ...")
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            try:
                # get aperture radius
                radius = self.scale*self.aperture_selector(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            except:
                # skip cameras with no sources
                continue
            
            print(f"[OPTICAM] Processing {fltr} files ...")
            
            times = [self.times[file] for file in self.camera_files[f"{fltr}"]]
            
            # get fluxes
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self._extract_aperture_fluxes_from_file, fltr=fltr, radius=radius), self.camera_files[f"{fltr}"]), total=len(self.camera_files[f"{fltr}"])))
            
            # unpack results
            fluxes, flux_errors, flags = zip(*results)
            times = [self.times[file] for file in self.camera_files[f"{fltr}"]]  # observation times
            
            print("[OPTICAM] Done.")
            print("[OPTICAM] Saving light curves ...")
            
            # for each source
            for i in tqdm(range(len(self.catalogs[f"{fltr}"]))):
                self._save_aperture_light_curve(times, fluxes, flux_errors, flags, fltr, i)
    
    def _extract_aperture_fluxes_from_file(self, file: str, fltr: str, radius: float) -> Tuple[List, List, str]:
        """
        Perform forced simple aperture photometry on a single image.
        
        Parameters
        ----------
        file : str
            The name of the file.
        fltr : str
            The filter.
        radius : float
            The aperture radius.
        
        Returns
        -------
        Tuple[List, List, str]
            The source fluxes, source flux errors, and image quality flag.
        """
        
        fluxes, flux_errors = [], []
        
        # if file does not have a transform and it's not the reference image
        if file not in self.transforms.keys() and file != self.camera_files[f"{fltr}"][0]:
            flag = "B"  # set flag to "B"
        else:
            flag = "A"  # set flag to "A"
            transform = self.transforms[file]  # get transform
        
        clean_data, error = self._get_background_subtracted_image_and_error(file)
        
        # for each source
        for i in range(len(self.catalogs[f"{fltr}"])):
            
            catalog_position = (self.catalogs[f"{fltr}"]["xcentroid"][i], self.catalogs[f"{fltr}"]["ycentroid"][i])
            
            # try to transform source position
            try:
                position = astroalign.matrix_transform(catalog_position, transform)[0]
            # if transform fails, use catalog position
            except:
                position = catalog_position
            
            # define aperture
            flux, flux_error = self._compute_aperture_flux(clean_data, error, position, radius)
            fluxes.append(flux)
            flux_errors.append(flux_error)
        
        return fluxes, flux_errors, flag
    
    def _get_background_subtracted_image_and_error(self, file: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        Get the background subtracted image and error for a given file.
        
        Parameters
        ----------
        file : str
            The name of the file.
        
        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            The background subtracted image and its error.
        """
        
        data = get_data(self.data_directory + file)
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
    
    def _save_aperture_light_curve(self, times: ArrayLike, fluxes: ArrayLike, flux_errors: ArrayLike, flags: ArrayLike,
                                   fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        times : ArrayLike
            The observation times.
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
            csvwriter.writerow(["MJD", "flux", "flux_error", "quality_flag"])
            
            # for each observation
            for i in range(len(self.camera_files[f"{fltr}"])):
                csvwriter.writerow([times[i], fluxes[i][source_index], flux_errors[i][source_index], flags[i]])
        
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
    
    def _extract_annulus_light_curves(self) -> None:
        """
        Perform forced aperture photometry with local background subtractions on all the images in out_directory.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"annulus_light_curves"):
            os.mkdir(self.out_directory + f"annulus_light_curves")
        
        print(f"[OPTICAM] Extracting annulus fluxes ...")
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            try:
                # get aperture radius
                radius = self.scale*self.aperture_selector(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            except:
                # skip cameras with no sources
                continue
            
            print(f"[OPTICAM] Processing {fltr} files ...")
            
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self._extract_annulus_fluxes_from_file, fltr=fltr, radius=radius), self.camera_files[f"{fltr}"]), total=len(self.camera_files[f"{fltr}"])))
            
            # unpack results
            fluxes, flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = zip(*results)
            times = [self.times[file] for file in self.camera_files[f"{fltr}"]]  # observation times
            
            print("[OPTICAM] Done.")
            print("[OPTICAM] Saving light curves ...")
            
            # for each source
            for i in range(len(self.catalogs[f"{fltr}"])):
                self._save_annulus_light_curve(times, fluxes, flux_errors, local_backgrounds,
                                               local_background_errors, local_backgrounds_per_pixel,
                                               local_background_errors_per_pixel, flags, fltr, i)
    
    def _extract_annulus_fluxes_from_file(self, file: str, fltr: str, radius: float) -> Tuple[List, List, List, List, List, List, str]:
        """
        Perform aperture photometry with local background subtractions on a single image.
        
        Parameters
        ----------
        file : str
            The name of the file.
        fltr : str
            The filter.
        radius : float
            The aperture radius.
        
        Returns
        -------
        Tuple[List, List, List, List, List, List, str]
            The source fluxes, source flux errors, local background, local background errors, local background per pixel,
            local background error per pixel, and image quality flag.
        """
        
        fluxes, flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
        
        # if file does not have a transform and it's not the reference image
        if file not in self.transforms.keys() and file != self.camera_files[f"{fltr}"][0]:
            flag = "B"  # set flag to "B"
        else:
            flag = "A"  # set flag to "A"
            transform = self.transforms[file]  # get transform
        
        data, error = self._get_image_and_error(file)
        
        # for each source
        for i in range(len(self.catalogs[f"{fltr}"])):
            
            catalog_position = (self.catalogs[f"{fltr}"]["xcentroid"][i], self.catalogs[f"{fltr}"]["ycentroid"][i])
            
            # try to transform source position
            try:
                position = astroalign.matrix_transform(catalog_position, transform)[0]
            # if transform fails, use catalog position
            except:
                position = catalog_position
            
            flux, flux_error, local_background, local_background_error, local_background_per_pixel, local_background_error_per_pixel = self._compute_annulus_flux(data, error, position, radius, self.r_in_scale*radius, self.r_out_scale*radius, self.local_background_method, self.local_background_sigma_clip)
            fluxes.append(flux)
            flux_errors.append(flux_error)
            local_backgrounds.append(local_background)
            local_background_errors.append(local_background_error)
            local_backgrounds_per_pixel.append(local_background_per_pixel)
            local_background_errors_per_pixel.append(local_background_error_per_pixel)
        
        return fluxes, flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flag

    def _get_image_and_error(self, file: str) -> Tuple[ArrayLike, ArrayLike]:
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
        error = np.sqrt(data*self.gains[file])  # Poisson noise
        
        return data, error
    
    @staticmethod
    def _compute_annulus_flux(data: ArrayLike, error: ArrayLike, position: ArrayLike, radius: float, r_in: float,
                              r_out: float, local_background_method: Literal['mean', 'median'],
                              sigma_clip: SigmaClip) -> Tuple[float, float, float, float, float, float]:
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
        r_in : float
            The annulus inner radius.
        r_out : float
            The annulus outer radius.
        local_background_method : Literal['mean', 'median']
            The method to use for calculating the local background.
        sigma_clip : SigmaClip
            The sigma clipper.
        
        Returns
        -------
        Tuple[float, float, float, float, float, float]
            The flux, flux error, local background, local background error, local background per pixel, and local
            background error per pixel.
        """
        
        # define aperture
        aperture = CircularAperture(position, r=radius)
        annulus_aperture = CircularAnnulus(position, r_in=r_in, r_out=r_out)
        aperstats = ApertureStats(data, annulus_aperture, error=error, sigma_clip=sigma_clip)
        aperture_area = aperture.area_overlap(data)
        
        # calculate local background per pixel
        if local_background_method == "median":
            local_background_per_pixel = aperstats.median
            local_background_error_per_pixel = aperstats.mad_std
        elif local_background_method == "mean":
            local_background_per_pixel = aperstats.mean
            local_background_error_per_pixel = aperstats.std
        
        # calculate total background in aperture
        total_bkg = local_background_per_pixel*aperture_area
        total_bkg_error = local_background_error_per_pixel*np.sqrt(aperture_area)
        
        phot_table = aperture_photometry(data, aperture, error=error)
        
        flux = phot_table["aperture_sum"].value[0] - total_bkg
        flux_error = np.sqrt(phot_table["aperture_sum_err"].value[0]**2 + total_bkg_error**2)
        local_background = total_bkg
        local_background_errors = total_bkg_error
        local_backgrounds_per_pixel = local_background_per_pixel
        local_background_errors_per_pixel = local_background_error_per_pixel
        
        return flux, flux_error, local_background, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel
    
    def _save_annulus_light_curve(self, times: ArrayLike, fluxes: ArrayLike, flux_errors: ArrayLike,
                                  local_backgrounds: ArrayLike, local_background_errors: ArrayLike,
                                  local_backgrounds_per_pixel: ArrayLike, local_background_errors_per_pixel: ArrayLike,
                                  flags: ArrayLike, fltr: str, source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        times : ArrayLike
            The observation times.
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
            csvwriter.writerow(["MJD", "flux", "flux_error", "local_background", "local_background_error",
                                "local_background_per_pixel", "local_background_error_per_pixel", "quality_flag"])
            for j in range(len(self.camera_files[f"{fltr}"])):
                csvwriter.writerow([times[j], fluxes[j][source_index], flux_errors[j][source_index],
                                    local_backgrounds[j][source_index], local_background_errors[j][source_index],
                                    local_backgrounds_per_pixel[j][source_index],
                                    local_background_errors_per_pixel[j][source_index], flags[j]])
        
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
    
    def _extract_aperture_and_annulus_light_curves(self) -> None:
        """
        Extract both simple aperture and local-background-subtracted aperture fluxes. This method is more efficient than
        calling _extract_aperture_light_curves() and _extract_annulus_light_curves() separately since it only opens the
        file once.
        """
        
        # create output directories if they do not exist
        if not os.path.isdir(self.out_directory + f"aperture_light_curves"):
            os.mkdir(self.out_directory + f"aperture_light_curves")
        if not os.path.isdir(self.out_directory + f"annulus_light_curves"):
            os.mkdir(self.out_directory + f"annulus_light_curves")
        
        print(f"[OPTICAM] Extracting aperture and annulus fluxes ...")
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            try:
                # get aperture radius
                radius = self.scale*self.aperture_selector(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            except:
                # skip cameras with no sources
                continue
            
            print(f"[OPTICAM] Processing {fltr} files ...")
            
            # get fluxes
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self._extract_aperture_and_annulus_fluxes_from_file, fltr=fltr, radius=radius), self.camera_files[f"{fltr}"]), total=len(self.camera_files[f"{fltr}"])))
            
            # unpack results
            aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flags = zip(*results)
            times = [self.times[file] for file in self.camera_files[f"{fltr}"]]  # observation times
            
            print("[OPTICAM] Done.")
            print("[OPTICAM] Saving light curves ...")
            
            # for each source
            for i in tqdm(range(len(self.catalogs[f"{fltr}"]))):
                self._save_aperture_light_curve(times, aperture_fluxes, aperture_flux_errors, flags, fltr, i)
                self._save_annulus_light_curve(times, annulus_fluxes, annulus_flux_errors, local_backgrounds,
                                               local_background_errors, local_backgrounds_per_pixel,
                                               local_background_errors_per_pixel, flags, fltr, i)
            
            print(f"[OPTICAM] Done.")
    
    def _extract_aperture_and_annulus_fluxes_from_file(self, file: str, fltr: str,radius: float) -> Tuple[List, List, List, List, List, List, List, List, str]:
        """
        Extract both simple aperture and local-background-subtracted aperture fluxes from a single image.
        
        Parameters
        ----------
        file : str
            The name of the file.
        fltr : str
            The filter.
        radius : float
            The aperture radius.
        
        Returns
        -------
        Tuple[List, List, List, List, List, List, List, List, str]
            The fluxes, flux errors, local-background-subtracted fluxes, local-background-subtracted flux errors, local
            backgrounds, local background errors, local backgrounds per pixel, local background errors per pixel, and
            image quality flag.
        """
        
        aperture_fluxes, aperture_flux_errors = [], []
        annulus_fluxes, annulus_flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
        
        # if file does not have a transform and it's not the reference image
        if file not in self.transforms.keys() and file != self.camera_files[f"{fltr}"][0]:
            flag = "B"  # set flag to "B"
        else:
            flag = "A"  # set flag to "A"
            transform = self.transforms[file]  # get transform
        
        data = get_data(self.data_directory + file)  # open image
        error = np.sqrt(data*self.gains[file])  # Poisson noise
        bkg = self.background(data)  # get background
        clean_data = data - bkg.background  # subtract background
        clean_error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])  # total error
        
        # for each source
        for i in range(len(self.catalogs[f"{fltr}"])):
            
            # get source catalog position
            catalog_position = (self.catalogs[f"{fltr}"]["xcentroid"][i], self.catalogs[f"{fltr}"]["ycentroid"][i])
            
            # get the aligned source position if possible
            try:
                position = astroalign.matrix_transform(catalog_position, transform)[0]
            except:
                position = catalog_position
            
            # get aperture flux
            aperture_flux, aperture_flux_error = self._compute_aperture_flux(clean_data, clean_error, position, radius)
            aperture_fluxes.append(aperture_flux)
            aperture_flux_errors.append(aperture_flux_error)
            
            # get aperture - annulus flux
            annulus_flux, annulus_flux_error, local_background, local_background_error, local_background_per_pixel, local_background_error_per_pixel = self._compute_annulus_flux(data, error, position, radius, self.r_in_scale*radius, self.r_out_scale*radius, self.local_background_method, self.local_background_sigma_clip)
            annulus_fluxes.append(annulus_flux)
            annulus_flux_errors.append(annulus_flux_error)
            local_backgrounds.append(local_background)
            local_background_errors.append(local_background_error)
            local_backgrounds_per_pixel.append(local_background_per_pixel)
            local_background_errors_per_pixel.append(local_background_error_per_pixel)
        
        return aperture_fluxes, aperture_flux_errors, annulus_fluxes, annulus_flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel, flag




    def photometry(self, phot_type: Literal['both', 'normal', 'optimal'] = 'both', tolerance: float = 5.) -> None:
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
        tolerance : float, optional
            The tolerance for source position matching in standard deviations (assuming a Gaussian PSF), by default 5.
            This parameter defines how far from the transformed catalog position a source can be while still being
            considered the same source. If the alignments are good and/or the field is crowded, consider reducing this
            value. For poor alignments and/or uncrowded fields, this value can be increased.
        """
        
        # determine which photometry function to use
        if phot_type == "normal":
            return self._extract_normal_light_curve(tolerance)
        elif phot_type == "optimal":
            return self._extract_optimal_light_curve(tolerance)
        elif phot_type == "both":
            self._extract_normal_and_optimal_light_curve(tolerance)
        else:
            raise ValueError(f"[OPTICAM] Photometry type {phot_type} not recognised.")
    
    def _extract_normal_light_curve(self, tolerance: float) -> None:
        """
        Extract the source fluxes from the images using simple aperture photometry. Unlike the forced photometry methods,
        this method requires fitting for the source positions in each image; as such, this method can be significantly
        slower. Moreover, this method can also misidentify sources if the field is crowded or the alignments are poor.
        
        Parameters
        ----------
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the alignments
            are good and the field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields,
            this value can be increased.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"normal_light_curves"):
            os.mkdir(self.out_directory + f"normal_light_curves")
        
        print(f"[OPTICAM] Extracting normal fluxes ...")
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            # get PSF parameters
            semimajor_sigma = self.aperture_selector(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            semiminor_sigma = self.aperture_selector(self.catalogs[f"{fltr}"]["semiminor_sigma"].value)
            
            print(f"[OPTICAM] Processing {fltr} files ...")
            
            # extract source light curve
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self._extract_normal_source_fluxes_from_file,
                                                      fltr=fltr, semimajor_sigma=semimajor_sigma,
                                                      semiminor_sigma=semiminor_sigma,
                                                      tolerance=tolerance),
                                              self.camera_files[f"{fltr}"]),
                                    total=len(self.camera_files[f"{fltr}"])))
            
            results = [result for result in results if result is not None]  # remove None values
            
            # unpack results
            times, fluxes, flux_errors, file_detections = zip(*results)
            
            print(f"[OPTICAM] Saving light curves ...")
            
            # for each source
            for i in tqdm(range(len(self.catalogs[f"{fltr}"]))):
                self._save_normal_light_curve(times, fluxes, flux_errors, fltr, i)
            
            detections = np.sum(file_detections, axis=0)  # sum detections across all files
            self._plot_number_of_detections_per_source(detections, fltr)  # plot number of detections per source
    
    def _extract_normal_source_fluxes_from_file(self, file: str, fltr: str, semimajor_sigma: float, 
                                                semiminor_sigma: float, tolerance: float) -> Tuple[float, float, float]:
        """
        Extract the source fluxes from an image using simple aperture photometry. Unlike the forced photometry methods,
        this method requires fitting for the source positions in each image; as such, this method is significantly
        slower. Moreover, this method can also misidentify sources if the field is crowded or the alignments are poor.
        
        Parameters
        ----------
        file : str
            The name of the image file.
        fltr : str
            The filter of the image.
        semimajor_sigma : float
            The semimajor axis of the (presumed 2D Gaussian) PSF.
        semiminor_sigma : float
            The semiminor axis of the (presumed 2D Gaussian) PSF.
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the source is
            further than this tolerance, it will be considered a different source. If the alignments are good and the
            field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields, this value can
            be increased.
        
        Returns
        -------
        Tuple[float, float, float]
            The observation time, source flux, and source flux error.
        """
        
        # if file does not have a transform, and it's not the reference image, skip it
        if file not in self.transforms.keys() and file != self.camera_files[f"{fltr}"][0]:
            return None
        
        fluxes, flux_errors = [], []
        detections = np.zeros(len(self.catalogs[f"{fltr}"]))
        
        # load image data
        data = get_data(self.data_directory + file)
        bkg = self.background(data)
        clean_data = data - bkg.background
        error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])
        
        # find sources in the image
        try:
            segment_map = self.finder(clean_data, self.threshold*bkg.background_rms)
        except:
            return None
        
        # create source catalog
        file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
        file_tbl = file_cat.to_table()
        
        # for each source
        for i in range(len(self.catalogs[f"{fltr}"])):
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
            flux, flux_error = self._compute_normal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[f"{fltr}"]["orientation"][i].value)
            fluxes.append(flux)
            flux_errors.append(flux_error)
        
        return self.times[file], fluxes, flux_errors, detections
    
    def _save_normal_light_curve(self, times: ArrayLike, fluxes: ArrayLike, flux_errors: ArrayLike, fltr: str,
                                 source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        times : ArrayLike
            The observation times.
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
            csvwriter.writerow(["MJD", "flux", "flux_error"])
            
            # for each observation in which a source was detected
            for i in range(len(times)):
                # if the source was detected
                if fluxes[i][source_index] is not None:
                    csvwriter.writerow([times[i], fluxes[i][source_index], flux_errors[i][source_index]])
        
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
    
    def _extract_optimal_light_curve(self, tolerance: float) -> None:
        """
        Use the optimal photometry method of Naylor 1998, MNRAS, 296, 339 to extract source fluxes from the images.
        Unlike the forced photometry methods, this method requires fitting for the source positions in each image; as
        such, this method can be significantly slower. Moreover, this method can also misidentify sources if the field
        is crowded or the alignments are poor.
        
        Parameters
        ----------
        tolerance : float
            The tolerance for source position matching in standard deviations. This parameter defines how far from the
            transformed catalog position a source can be while still being considered the same source. If the source is
            further than this tolerance, it will be considered a different source. If the alignments are good and the
            field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields, this value can
            be increased.
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"optimal_light_curves"):
            os.mkdir(self.out_directory + f"optimal_light_curves")
        
        print(f"[OPTICAM] Extracting optimal fluxes ...")
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            # get PSF parameters
            semimajor_sigma = self.aperture_selector(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            semiminor_sigma = self.aperture_selector(self.catalogs[f"{fltr}"]["semiminor_sigma"].value)
            
            print(f"[OPTICAM] Processing {fltr} files ...")
            
            # extract source light curve
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self._extract_optimal_source_fluxes_from_file,
                                                      fltr=fltr, semimajor_sigma=semimajor_sigma,
                                                      semiminor_sigma=semiminor_sigma,
                                                      tolerance=tolerance),
                                              self.camera_files[f"{fltr}"]),
                                    total=len(self.camera_files[f"{fltr}"])))
            
            results = [result for result in results if result is not None]  # remove None values
            
            # unpack results
            times, fluxes, flux_errors, file_detections = zip(*results)
            
            print(f"[OPTICAM] Saving light curves ...")
            
            # for each source
            for i in tqdm(range(len(self.catalogs[f"{fltr}"]))):
                self._save_optimal_light_curve(times, fluxes, flux_errors, fltr, i)
            
            detections = np.sum(file_detections, axis=0)  # sum detections across all files
            self._plot_number_of_detections_per_source(detections, fltr)  # plot number of detections per source
    
    def _extract_optimal_source_fluxes_from_file(self, file: str, fltr: str, semimajor_sigma: float,
                                                 semiminor_sigma: float, tolerance: float) -> Tuple[float, float, float]:
        """
        Use the optimal photometry method of Naylor 1998, MNRAS, 296, 339 to extract the source flux from an image.
        Unlike the forced photometry methods, this method requires fitting for the source positions in each image; as
        such, this method can be significantly slower. Moreover, this method can also misidentify sources if the field
        is crowded or the alignments are poor.
        
        Parameters
        ----------
        file : str
            The name of the image file.
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
        
        Returns
        -------
        Tuple[float, float, float]
            The observation time, source flux, and source flux error.
        """
        
        # if file does not have a transform, and it's not the reference image, skip it
        if file not in self.transforms.keys() and file != self.camera_files[f"{fltr}"][0]:
            return None
        
        fluxes, flux_errors = [], []
        detections = np.zeros(len(self.catalogs[f"{fltr}"]))
        
        # load image data
        data = get_data(self.data_directory + file)
        bkg = self.background(data)
        clean_data = data - bkg.background
        error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])
        
        # find sources in the image
        try:
            segment_map = self.finder(clean_data, self.threshold*bkg.background_rms)
        except:
            return None
        
        # create source catalog
        file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
        file_tbl = file_cat.to_table()
        
        # for each source
        for i in range(len(self.catalogs[f"{fltr}"])):
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
            flux, flux_error = self._compute_optimal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[f"{fltr}"]["orientation"][i].value)
            fluxes.append(flux)
            flux_errors.append(flux_error)
        
        return self.times[file], fluxes, flux_errors, detections
    
    def _save_optimal_light_curve(self, times: ArrayLike, fluxes: ArrayLike, flux_errors: ArrayLike, fltr: str,
                                  source_index: int) -> None:
        """
        Plot and save the light curve.
        
        Parameters
        ----------
        times : ArrayLike
            The observation times.
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
            csvwriter.writerow(["MJD", "flux", "flux_error"])
            
            # for each observation in which a source was detected
            for i in range(len(times)):
                # if the source was detected
                if fluxes[i][source_index] is not None:
                    csvwriter.writerow([times[i], fluxes[i][source_index], flux_errors[i][source_index]])
        
        df = pd.read_csv(self.out_directory + f"optimal_light_curves/{fltr}_source_{source_index + 1}.csv")
        
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
        fig.savefig(self.out_directory + f"optimal_light_curves/{fltr}_source_{source_index + 1}.png")
        
        plt.close(fig)
    
    def _extract_normal_and_optimal_light_curve(self, tolerance: float) -> None:
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
        """
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"normal_light_curves"):
            os.mkdir(self.out_directory + f"normal_light_curves")
        if not os.path.isdir(self.out_directory + f"optimal_light_curves"):
            os.mkdir(self.out_directory + f"optimal_light_curves")
        
        print(f"[OPTICAM] Extracting normal and optimal fluxes ...")
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            # get PSF parameters
            semimajor_sigma = self.aperture_selector(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            semiminor_sigma = self.aperture_selector(self.catalogs[f"{fltr}"]["semiminor_sigma"].value)
            
            print(f"[OPTICAM] Processing {fltr} files ...")
            
            # extract source light curve
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self._extract_normal_and_optimal_source_fluxes_from_file,
                                                      fltr=fltr, semimajor_sigma=semimajor_sigma,
                                                      semiminor_sigma=semiminor_sigma,
                                                      tolerance=tolerance),
                                              self.camera_files[f"{fltr}"]),
                                    total=len(self.camera_files[f"{fltr}"])))
            
            results = [result for result in results if result is not None]  # remove None values
            
            # unpack results
            times, normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, file_detections = zip(*results)
            
            print(f"[OPTICAM] Saving light curves ...")
            
            # for each source
            for i in tqdm(range(len(self.catalogs[f"{fltr}"]))):
                self._save_normal_light_curve(times, normal_fluxes, normal_flux_errors, fltr, i)
                self._save_optimal_light_curve(times, optimal_fluxes, optimal_flux_errors, fltr, i)
            
            detections = np.sum(file_detections, axis=0)  # sum detections across all files
            self._plot_number_of_detections_per_source(detections, fltr)  # plot number of detections per source
    
    def _extract_normal_and_optimal_source_fluxes_from_file(self, file: str, fltr: str, semimajor_sigma: float,
                                                            semiminor_sigma: float, tolerance: float) -> Tuple[float, float, float]:
        """
        Extract both normal and optimal source fluxes from an image. This method is more efficient than calling
        _extract_normal_light_curve() and _extract_optimal_light_curve() separately since it only opens the file once.
        
        Parameters
        ----------
        file : str
            The name of the image file.
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
        
        Returns
        -------
        Tuple[float, float, float]
            The observation time, source flux, and source flux error.
        """
        
        # if file does not have a transform, and it's not the reference image, skip it
        if file not in self.transforms.keys() and file != self.camera_files[f"{fltr}"][0]:
            return None
        
        normal_fluxes, normal_flux_errors = [], []
        optimal_fluxes, optimal_flux_errors = [], []
        detections = np.zeros(len(self.catalogs[f"{fltr}"]))
        
        # load image data
        data = get_data(self.data_directory + file)
        bkg = self.background(data)
        clean_data = data - bkg.background
        error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])
        
        # find sources in the image
        try:
            segment_map = self.finder(clean_data, self.threshold*bkg.background_rms)
        except:
            return None
        
        # create source catalog
        file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
        file_tbl = file_cat.to_table()
        
        # for each source
        for i in range(len(self.catalogs[f"{fltr}"])):
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
            normal_flux, normal_flux_error = self._compute_normal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[f"{fltr}"]["orientation"][i].value)
            normal_fluxes.append(normal_flux)
            normal_flux_errors.append(normal_flux_error)            
            optimal_flux, optimal_flux_error = self._compute_optimal_flux(clean_data, error, position, semimajor_sigma, semiminor_sigma, self.catalogs[f"{fltr}"]["orientation"][i].value)
            optimal_fluxes.append(optimal_flux)
            optimal_flux_errors.append(optimal_flux_error)
        
        return self.times[file], normal_fluxes, normal_flux_errors, optimal_fluxes, optimal_flux_errors, detections

    def _compute_normal_flux(self, clean_data: ArrayLike, error: ArrayLike, position: ArrayLike, semimajor_sigma: float,
                             semiminor_sigma: float, orientation: float) -> Tuple[float, float]:
        """
        Compute the flux at a given position using simple aperture photometry.
        
        Parameters
        ----------
        clean_data : ArrayLike
            The background subtracted image.
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
        
        Returns
        -------
        Tuple[float, float]
            The flux and its error.
        """
        
        aperture = EllipticalAperture(position, semimajor_sigma, semiminor_sigma, orientation)  # define aperture
        phot_table = aperture_photometry(clean_data, aperture, error=error)  # perform aperture photometry
        
        return phot_table["aperture_sum"].value[0], phot_table["aperture_sum_err"].value[0]

    def _compute_optimal_flux(self, clean_data: ArrayLike, error: ArrayLike, position: ArrayLike, semimajor_sigma: float,
                                semiminor_sigma: float, orientation: float) -> Tuple[float, float]:
        """
        Compute the flux at a given position using the optimal photometry method of Naylor 1998, MNRAS, 296, 339.
        
        Parameters
        ----------
        clean_data : ArrayLike
            The background subtracted image.
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
        
        Returns
        -------
        Tuple[float, float]
            The flux and its error.
        """
        
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
        catalog_position = (self.catalogs[f"{fltr}"]["xcentroid"][source_index], self.catalogs[f"{fltr}"]["ycentroid"][source_index])
        
        # if file is the reference image
        if file == self.camera_files[f"{fltr}"][0]:
            # use the catalog position as the initial position
            initial_position = catalog_position
        else:
            # use the transformed catalog position as the initial position
            initial_position = astroalign.matrix_transform(catalog_position, self.transforms[file])[0]
        
        # get positions of sources
        positions = np.array([[file_tbl["xcentroid"][i], file_tbl["ycentroid"][i]] for i in range(len(file_tbl))])
        
        # get distances between sources and initial position
        distances = np.sqrt((positions[:, 0] - initial_position[0])**2 + (positions[:, 1] - initial_position[1])**2)
        
        # if the closest source is further than the specified tolerance
        if np.min(distances) > tolerance*np.sqrt(self.catalogs[f"{fltr}"]["semimajor_sigma"][source_index].value**2 + self.catalogs[f"{fltr}"]["semiminor_sigma"][source_index].value**2):
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
            for i in range(len(self.catalogs[f"{fltr}"])):
                csvwriter.writerow([i + 1, detections[i]])
        
        fig, ax = plt.subplots(tight_layout=True)
        
        ax.bar(np.arange(len(self.catalogs[f"{fltr}"])) + 1, detections, color="none", edgecolor="black")
        ax.axhline(len(self.camera_files[f"{fltr}"]), color="red", linestyle="--", lw=1)
        
        ax.set_xlabel("Source")
        ax.set_ylabel("Number of detections")
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
        
        fig.savefig(self.out_directory + f"diag/{fltr}_observations.png")
        
        if self.show_plots:
            plt.show(fig)
        else:
            plt.close(fig)
