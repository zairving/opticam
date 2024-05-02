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
from photutils.background import Background2D
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

from opticam_new.helpers import get_data
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
        aperture_selector: Literal["default", "max", "min", "median"] = "default",
        scale: float = 5,
        r_in_scale: float = 1,
        r_out_scale: float = 2,
        local_background_method: Literal["mean", "median"] = "mean",
        local_background_sigma_clip: SigmaClip = SigmaClip(sigma=3, maxiters=10),
        number_of_processors: int = cpu_count(),
        show_plots: bool = True,
        ) -> None:
        """
        Helper class for working with OPTICAM data.

        Parameters
        ----------
        data_directory : str
            The directory containing the data files.
        out_directory : str
            The directory to save the output files.
        sigma : float, optional
            The sigma value for sigma clipping, by default 3.
        threshold : float, optional
            The threshold value for source detection, by default 5. This value is multiplied by the background RMS to
            determine the threshold for source detection.
        background : Callable, optional
            The background calculator. If None (default), `photutils.background.Background2D` is used with `box_size`
            set based on the binning of the images. `photutils.background.Background2D` uses the SExtractorBackground
            and StdBackgroundRMS estimators by default. If a custom background calculator is provided, it must take an
            image as input and return a `photutils.background.Background2D` object.
        finder : Callable, optional
            The source finder. If None (default), `photutils.segmentation.SourceFinder` is used with `npixels` set based
            on the binning of the images. If a custom source finder is provided, it must take an image as input and
            return a `photutils.segmentation.SegementationImage` object.
        scale : float, optional
            The scale factor for the source apertures, by default 5. Sources are modelled as bivariate Gaussians, so a
            scale factor of 5 means that the aperture will be 5 times the standard deviation of the Gaussian along its
            major axis.
        number_of_processors : int, optional
            The number of processors to use for parallel processing, by default cpu_count(). Increasing this number will
            increase performance at the cost of increased memory usage. To avoid prohibitive memory usage, some methods
            are not parallelised.
        show_plots : bool, optional
            Whether to show plots, by default True.
        """
        
        assert aperture_selector in ["default", "max", "min", "median"], "Aperture selection must be \"default\" \"max\", \"min\", or \"median\"."
        if aperture_selector == "default":
            self.aperture_selector = self._default_aperture_selector
        elif aperture_selector == "max":
            self.aperture_selector = np.max
        elif aperture_selector == "min":
            self.aperture_selector = np.min
        elif aperture_selector == "median":
            self.aperture_selector = np.median
        
        assert local_background_method in ["mean", "median"], "Local background method must be 'mean' or 'median'."
        self.local_background_method = local_background_method
        
        # format data directory
        self.data_directory = data_directory
        if self.data_directory[-1] != "/":
            self.data_directory += "/"
        
        self.file_names = sorted(os.listdir(self.data_directory))  # get list of file names
        
        print("[OPTICAM] Scanning files ...")
        
        # check binning and parse observation times
        self.times = {}  # file name : file MJD
        self.gains = {}
        binning = []
        self.camera_files = {}
        for file_name in tqdm(self.file_names):
            with fits.open(self.data_directory + file_name) as hdul:
                # get file binning
                if hdul[0].header["BINNING"] not in binning:
                    binning.append(hdul[0].header["BINNING"])
                
                self.gains.update({file_name: hdul[0].header["GAIN"]})  # get gain (for error calculation)
                
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
                    raise KeyError("[OPTICAM] Could not find GPSTIME or UT key in header.")
                
                # separate files by filter
                fltr = hdul[0].header["FILTER"]
                if f"{fltr}-band" not in self.camera_files.keys():
                    self.camera_files.update({f"{fltr}-band": []})
                self.camera_files[f"{fltr}-band"].append(file_name)
                
                self.times.update({file_name: mjd})  # get file MJD
        
        for key in list(self.camera_files.keys()):
            self.camera_files[key].sort(key=lambda x: self.times[x])  # sort files by time
        
        print("[OPTICAM] Done.")
        
        # if binning is not consistent, raise error
        if len(binning) > 1:
            raise ValueError("[OPTICAM] All images must have the same binning.")
        else:
            binning = binning[0]
        
        print(f"[OPTICAM] Binning: {binning}")
        
        if len(list(self.camera_files.keys())) > 3:
            raise ValueError("[OPTICAM] More than 3 filters found.")
        
        # format output directory
        self.out_directory = out_directory
        if self.out_directory[-1] != "/":
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
        
        with open(self.out_directory + "misc/filters.txt", "w") as file:
            file.write("\n".join(list(self.camera_files.keys())))
        
        self.t_ref = min(list(self.times.values()))  # get reference time
        with open(out_directory + "misc/earliest_observation_time.txt", "w") as file:
            file.write(str(self.t_ref))
        
        # set parameters
        self.scale = scale
        self.r_in_scale = r_in_scale
        self.r_out_scale = r_out_scale
        self.threshold = threshold
        self.local_background_sigma_clip = local_background_sigma_clip
        
        # set parameters
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        # define colours for circling sources in catalogs
        self.colours = list(mcolors.TABLEAU_COLORS.keys())
        
        # remove colours that are hard to see on a dark background
        self.colours.pop(self.colours.index("tab:brown"))
        self.colours.pop(self.colours.index("tab:gray"))
        self.colours.pop(self.colours.index("tab:purple"))
        self.colours.pop(self.colours.index("tab:blue"))
        
        # define parameters dictionary
        param_dict = {
            "threshold": threshold,
            "aperture scale": scale,
            "annulus r_in scale": r_in_scale,
            "annulus r_out scale": r_out_scale,
            "aperture selector": aperture_selector,
            "local background method": local_background_method,
        }
        
        for key, value in self.local_background_sigma_clip.__dict__.items():
            if not key.startswith("_"):
                param_dict["local background SigmaClip " + str(key)] = value

        param_dict.update({"number of files": len(self.file_names)})
        param_dict.update({f"number of {fltr} files": len(self.camera_files[f"{fltr}"]) for fltr in list(self.camera_files.keys())})
        
        # write input parameters to file
        with open(self.out_directory + "misc/reducer_input.json", "w") as file:
            json.dump(param_dict, file, indent=4)
        
        # define background calculator
        if background is None:
            if binning not in ["4x4", "3x3", "2x2", "1x1"]:
                raise ValueError(f"[OPTICAM] Binning {binning} is not (yet) supported! Supported binning values are '4x4', '3x3', '2x2', and '1x1'.")
            box_size = 16 if binning == "4x4" else 22 if binning == "3x3" else 32 if binning == "2x2" else 64  # set box_size based on binning
            self.background = Background(box_size=box_size)
        else:
            self.background = background
        
        # write  background input parameters to file
        try:
            with open(self.out_directory + "misc/background_input.json", "w") as file:
                json.dump(self.background.get_input_dict(), file, indent=4)
        except:
            warnings.warn("[OPTICAM] Could not write background input parameters to file.")
        
        # define source finder
        if finder is None:
            if binning not in ["4x4", "3x3", "2x2", "1x1"]:
                raise ValueError(f"[OPTICAM] Binning {binning} is not (yet) supported! Supported binning values are '4x4', '3x3', '2x2', and '1x1'.")
            npixels = 25 if binning == "4x4" else 33 if binning == "3x3" else 50 if binning == "2x2" else 100
            border_width = 16 if binning == "4x4" else 22 if binning == "3x3" else 32 if binning == "2x2" else 64
            self.finder = Finder(npixels=npixels, border_width=border_width)
        else:
            self.finder = finder
        
        # write finder input parameters to file
        try:
            with open(self.out_directory + "misc/finder_input.json", "w") as file:
                json.dump(self.finder.get_input_dict(), file, indent=4)
        except:
            warnings.warn("[OPTICAM] Could not write finder input parameters to file")
        
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
    
    def initialise_catalogs(self) -> None:
        """
        Initialise the source catalogs for each camera.
        """
        
        ## TODO: combine background
        
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
            
            print(f"[OPTICAM] Aligning {fltr} images ...")
            
            reference_image = get_data(self.data_directory + self.camera_files[f"{fltr}"][0])  # get reference image
            
            stacked_images = reference_image.copy()  # initialise stacked images as reference image
            
            cat_ax[list(self.camera_files.keys()).index(fltr)].set_title(f"{fltr}")
            cat_ax[list(self.camera_files.keys()).index(fltr)].set_xlabel("X")
            cat_ax[list(self.camera_files.keys()).index(fltr)].set_ylabel("Y")
            
            transforms = {}  # define transforms as empty dictionary
            unaligned_files = []
            
            background_median.update({f"{fltr}": []})
            background_rms.update({f"{fltr}": []})
            
            for i in tqdm(range(len(self.camera_files[f"{fltr}"]))):
                data = get_data(self.data_directory + self.camera_files[f"{fltr}"][i])  # get image data
                
                # try to align image
                try:
                    transform = astroalign.find_transform(reference_image, data)[0].params.tolist()  # align image w.r.t reference image
                    transforms.update({self.camera_files[f"{fltr}"][i]: transform})  # store transform in dictionary
                    stacked_images += astroalign.apply_transform(SimilarityTransform(transform), data, reference_image)[0]  # align and stack image
                except:
                    unaligned_files.append(self.camera_files[f"{fltr}"][i])  # store unaligned file in list
                
                # get background
                background = self.background(data)
                background_median[f"{fltr}"].append(background.background_median)
                background_rms[f"{fltr}"].append(background.background_rms_median)
            
            self.transforms.update(transforms)
            self.unaligned_files += unaligned_files
            
            print(f"[OPTICAM] Done. {len(unaligned_files)} image(s) could not be aligned.")
            
            # remove background from stacked images
            stacked_bkg = self.background(stacked_images)
            stacked_images -= stacked_bkg.background
            
            bkg_axs[list(self.camera_files.keys()).index(fltr)].imshow(stacked_images, origin="lower", cmap="Greys_r",
                                                                       interpolation="nearest",
                                                                       norm=simple_norm(stacked_images, stretch="log"))
            
            stacked_bkg.plot_meshes(ax=bkg_axs[list(self.camera_files.keys()).index(fltr)], outlines=True, marker='.',
                                    color='cyan', alpha=0.3)
            
            bkg_axs[list(self.camera_files.keys()).index(fltr)].set_title(f"{fltr}")
            
            try:
                # identify sources in stacked images
                segment_map = self.finder(stacked_images, self.threshold*stacked_bkg.background_rms)
            except:
                print(f"[OPTICAM] No sources found in {fltr}.")
                continue
            
            # create catalog of sources in stacked images
            self.catalogs.update({f"{fltr}": SourceCatalog(stacked_images, segment_map, background=stacked_bkg.background).to_table()})
            # write catalog to file
            self.catalogs[f"{fltr}"].write(self.out_directory + f"cat/{fltr}_catalog.ecsv", format="ascii.ecsv",
                                            overwrite=True)
            
            # plot stacked images
            cat_ax[list(self.camera_files.keys()).index(fltr)].imshow(stacked_images, origin="lower", cmap="Greys_r", interpolation="nearest",
                                      norm=simple_norm(stacked_images, stretch="log"))

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
        
        cat_fig.savefig(self.out_directory + "cat/catalogs.png")
        bkg_fig.savefig(self.out_directory + "diag/background_meshes.png")
        
        self._plot_time_between_files()
        self._plot_backgrounds(background_median, background_rms)
        
        # save transforms to file
        with open(self.out_directory + "cat/transforms.json", "w") as file:
            json.dump(self.transforms, file, indent=4)
        
        if self.show_plots:
            plt.show(cat_fig)
        else:
            plt.close(cat_fig)
    
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
            
            bin_edges = np.arange(0, int(dt.max()) + 2, 1)
            
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
    
    def create_gifs(self, keep_frames: bool = True) -> None:
        """
        Create alignment gifs for each camera.

        Parameters
        ----------
        keep_frames : bool, optional
            Whether to save the GIF frames in out_directory/diag, by default True. If False, the frames will be deleted
            after the GIF is saved. When reducing lots of images, it is recommended to set this to False to reduce disk
            usage.
        """
        
        print("[OPTICAM] Creating gifs ...")
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            else:
                if not os.path.isdir(self.out_directory + f"diag/{fltr}_gif_frames"):
                    os.mkdir(self.out_directory + f"diag/{fltr}_gif_frames")
            
            print(f"[OPTICAM] Creating {fltr} frames ...")
            
            # # for each file
            # for file in tqdm(self.camera_files[f"{fltr}"]):
                
            #     # create GIF frame
            #     self._create_gif_frame(file, fltr)
            
            # create GIF frames
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self._create_gif_frame, fltr=fltr), self.camera_files[f"{fltr}"]),
                               total=len(self.camera_files[f"{fltr}"])))
            
            print(f"[OPTICAM] Done.")
            print(f"[OPTICAM] Compiling frames ...")
            
            # save GIF
            self._create_gif(fltr, keep_frames)
            
            print(f"[OPTICAM] Done.")
    
    def _create_gif(self, fltr: str, keep_frames: bool) -> None:
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
            for file in self.camera_files[f"{fltr}"]:
                try:
                    os.remove(self.out_directory + f"{file.split(".")[0]}.png")
                except:
                    pass
    
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
    
    def _plot_backgrounds(self, background_median: Dict[str, List], background_rms: Dict[str, List]) -> None:
        """
        Plot the time-varying background for each camera.
        """
        
        fig, axs = plt.subplots(nrows=2, ncols=3, tight_layout=True, sharex=True, figsize=(2*6.4, 2*4.8))
        
        # for each camera
        for fltr in list(self.camera_files.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            times = np.array([self.times[file] for file in self.camera_files[f"{fltr}"]])
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

        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    def forced_photometry(self, phot_type: Literal["aperture", "annulus", "both"] = "both") -> None:
        """
        Perform forced photometry on the images to extract source fluxes.
        
        Parameters
        ----------
        phot_type : Literal["aperture", "annulus", "both"], optional
            The type of photometry to perform, by default "both". If "aperture", only aperture photometry is performed.
            If "annulus", only annulus photometry is performed. If "both", both aperture and annulus photometry are
            performed simultaneously (much more efficiently that performing both separately).
        
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
    
    def _extract_aperture_light_curves(self):
        
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
            fluxes, flux_errors, flags = [], [], []
            
            for file in tqdm(self.camera_files[f"{fltr}"]):
                
                if file in self.transforms.keys():
                    flags.append("A")
                else:
                    flags.append("B")
                
                file_fluxes, file_flux_errors = self.__extract_aperture_light_curves(file, fltr, radius)
                fluxes.append(file_fluxes)
                flux_errors.append(file_flux_errors)
            
            # for each source
            for i in range(len(self.catalogs[f"{fltr}"])):
                
                # save source light curve to file
                with open(self.out_directory + f"aperture_light_curves/{fltr}_source_{i + 1}.csv", "w") as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(["MJD", "flux", "flux_error", "quality_flag"])
                    for j in range(len(self.camera_files[f"{fltr}"])):
                        csvwriter.writerow([times[j], fluxes[j][i], flux_errors[j][i], flags[j]])
                
                # load light curve from file
                df = pd.read_csv(self.out_directory + f"aperture_light_curves/{fltr}_source_{i + 1}.csv")
                
                aligned_mask = df["quality_flag"] == "A"
                
                # reformat MJD to seconds from first observation
                df["MJD"] -= self.t_ref
                df["MJD"] *= 86400
                
                fig, ax = plt.subplots(tight_layout=True)
                
                ax.errorbar(df["MJD"].values[aligned_mask], df["flux"].values[aligned_mask], yerr=df["flux_error"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                ax.errorbar(df["MJD"].values[~aligned_mask], df["flux"].values[~aligned_mask], yerr=df["flux_error"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
                
                ax.set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
                ax.set_ylabel("Flux [counts]")
                ax.set_title(f"{fltr} Source {i + 1}")
                
                ax.minorticks_on()
                ax.tick_params(which="both", direction="in", top=True, right=True)
                
                # save light curve plot to file
                fig.savefig(self.out_directory + f"aperture_light_curves/{fltr}_source_{i + 1}.png")
                
                plt.close(fig)
    
    def __extract_aperture_light_curves(self, file, fltr, radius):
        
        fluxes, flux_errors = [], []
        
        # if file is not the reference image, try to retrieve the transform
        if file != self.camera_files[f"{fltr}"][0]:
            try:
                # if transform exists, use it and set quality flag to "A"
                transform = self.transforms[file]
            except:
                pass
        
        data = get_data(self.data_directory + file)
        bkg = self.background(data)
        clean_data = data - bkg.background
        error = calc_total_error(clean_data, bkg.background_rms, self.gains[file])
        
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
            aperture = CircularAperture(position, r=radius)
            phot_table = aperture_photometry(clean_data, aperture, error=error)
            fluxes.append(phot_table["aperture_sum"].value[0])
            flux_errors.append(phot_table["aperture_sum_err"].value[0])
        
        return fluxes, flux_errors
    
    def _extract_annulus_light_curves(self):
        
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
            
            fluxes, flux_errors = [], []
            local_backgrounds, local_background_errors = [], []
            local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
            flags = []
            times = [self.times[file] for file in self.camera_files[f"{fltr}"]]
            
            for file in tqdm(self.camera_files[f"{fltr}"]):
                
                if file in self.transforms.keys():
                    flags.append("A")
                else:
                    flags.append("B")

                file_fluxes, file_flux_errors, file_local_backgrounds, file_local_background_errors, \
                file_local_backgrounds_per_pixel, file_local_background_errors_per_pixel \
                = self.__extract_annulus_light_curves(file, fltr, radius)
                
                fluxes.append(file_fluxes)
                flux_errors.append(file_flux_errors)
                local_backgrounds.append(file_local_backgrounds)
                local_background_errors.append(file_local_background_errors)
                local_backgrounds_per_pixel.append(file_local_backgrounds_per_pixel)
                local_background_errors_per_pixel.append(file_local_background_errors_per_pixel)
            
            # for each source
            for i in range(len(self.catalogs[f"{fltr}"])):
                
                # save source light curve to file
                with open(self.out_directory + f"annulus_light_curves/{fltr}_source_{i + 1}.csv", "w") as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(["MJD", "flux", "flux_error", "local_background", "local_background_error", "local_background_per_pixel", "local_background_error_per_pixel", "quality_flag"])
                    for j in range(len(self.camera_files[f"{fltr}"])):
                        csvwriter.writerow([times[j], fluxes[j][i], flux_errors[j][i], local_backgrounds[j][i], local_background_errors[j][i], local_backgrounds_per_pixel[j][i], local_background_errors_per_pixel[j][i], flags[j]])
                
                # load light curve from file
                df = pd.read_csv(self.out_directory + f"annulus_light_curves/{fltr}_source_{i + 1}.csv")
                
                aligned_mask = df["quality_flag"] == "A"
                
                # reformat MJD to seconds from first observation
                df["MJD"] -= self.t_ref
                df["MJD"] *= 86400
                
                fig, ax = plt.subplots(nrows=3, tight_layout=True, figsize=(6.4, 2*4.8), sharex=True, gridspec_kw={"hspace": 0})
                
                ax[0].errorbar(df["MJD"].values[aligned_mask], df["flux"].values[aligned_mask], yerr=df["flux_error"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                ax[0].errorbar(df["MJD"].values[~aligned_mask], df["flux"].values[~aligned_mask], yerr=df["flux_error"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
                ax[0].set_ylabel("Flux [counts]")
                ax[0].set_title(f"{fltr} Source {i + 1}")
                
                ax[1].errorbar(df["MJD"].values[aligned_mask], df["local_background_per_pixel"].values[aligned_mask], yerr=df["local_background_error_per_pixel"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                ax[1].errorbar(df["MJD"].values[~aligned_mask], df["local_background_per_pixel"].values[~aligned_mask], yerr=df["local_background_error_per_pixel"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
                ax[1].set_ylabel("Local background [cts/pixel]")
                
                ax[2].plot(df["MJD"].values[aligned_mask], df["flux"].values[aligned_mask]/df["local_background"].values[aligned_mask], "k.", ms=2)
                ax[2].plot(df["MJD"].values[~aligned_mask], df["flux"].values[~aligned_mask]/df["local_background"].values[~aligned_mask], "r.", ms=2, alpha=.2)
                ax[2].set_ylabel("SNR")
                ax[2].set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
                
                for _ in ax:
                    _.minorticks_on()
                    _.tick_params(which="both", direction="in", top=True, right=True)
                
                # save light curve plot to file
                fig.savefig(self.out_directory + f"annulus_light_curves/{fltr}_source_{i + 1}.png")
                
                plt.close(fig)
    
    def __extract_annulus_light_curves(self, file, fltr, radius):
        
        fluxes, flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
        
        if file != self.camera_files[f"{fltr}"][0]:
            try:
                transform = self.transforms[file]
            except:
                pass
        
        data = get_data(self.data_directory + file)
        error = np.sqrt(data*self.gains[file])  # Poisson noise
        
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
            aperture = CircularAperture(position, r=radius)
            annulus_aperture = CircularAnnulus(position, r_in=self.r_in_scale*radius, r_out=self.r_out_scale*radius)
            aperstats = ApertureStats(data, annulus_aperture, error=error, sigma_clip=SigmaClip(sigma=3, maxiters=10))
            aperture_area = aperture.area_overlap(data)
            
            # calculate local background per pixel
            if self.local_background_method == "median":
                local_background_per_pixel = aperstats.median
                local_background_error_per_pixel = aperstats.mad_std
            elif self.local_background_method == "mean":
                local_background_per_pixel = aperstats.mean
                local_background_error_per_pixel = aperstats.std
            
            # calculate total background in aperture
            total_bkg = local_background_per_pixel*aperture_area
            total_bkg_error = local_background_error_per_pixel*np.sqrt(aperture_area)
            
            phot_table = aperture_photometry(data, aperture, error=error)
            
            fluxes.append(phot_table["aperture_sum"].value[0] - total_bkg)
            flux_errors.append(np.sqrt(phot_table["aperture_sum_err"].value[0]**2 + total_bkg_error**2))
            local_backgrounds.append(total_bkg)
            local_background_errors.append(total_bkg_error)
            local_backgrounds_per_pixel.append(local_background_per_pixel)
            local_background_errors_per_pixel.append(local_background_error_per_pixel)
        
        return fluxes, flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel
    
    def _extract_aperture_and_annulus_light_curves(self):
        
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
            
            fluxes, flux_errors = [], []
            local_backgrounds, local_background_errors = [], []
            local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
            flags = []
            times = [self.times[file] for file in self.camera_files[f"{fltr}"]]
            
            for file in tqdm(self.camera_files[f"{fltr}"]):
                
                if file in self.transforms.keys():
                    flags.append("A")
                else:
                    flags.append("B")

                file_fluxes, file_flux_errors, file_local_backgrounds, file_local_background_errors, \
                file_local_backgrounds_per_pixel, file_local_background_errors_per_pixel \
                = self.__extract_aperture_and_annulus_light_curves(file, fltr, radius)
                
                fluxes.append(file_fluxes)
                flux_errors.append(file_flux_errors)
                local_backgrounds.append(file_local_backgrounds)
                local_background_errors.append(file_local_background_errors)
                local_backgrounds_per_pixel.append(file_local_backgrounds_per_pixel)
                local_background_errors_per_pixel.append(file_local_background_errors_per_pixel)
            
            # for each source
            for i in range(len(self.catalogs[f"{fltr}"])):
                
                # save aperture light curve to file
                with open(self.out_directory + f"aperture_light_curves/{fltr}_source_{i + 1}.csv", "w") as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(["MJD", "flux", "flux_error", "quality_flag"])
                    for j in range(len(fluxes)):
                        csvwriter.writerow([times[j], fluxes[j][i], flux_errors[j][i], flags[j]])
                
                # save annulus light curve to file
                with open(self.out_directory + f"annulus_light_curves/{fltr}_source_{i + 1}.csv", "w") as file:
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(["MJD", "flux", "flux_error", "local_background", "local_background_error", "local_background_per_pixel", "local_background_error_per_pixel", "quality_flag"])
                    for j in range(len(self.camera_files[f"{fltr}"])):
                        csvwriter.writerow([times[j], fluxes[j][i] - local_backgrounds[j][i],
                                            np.sqrt(flux_errors[j][i]**2 + local_background_errors[j][i]**2),
                                            local_backgrounds[j][i], local_background_errors[j][i],
                                            local_backgrounds_per_pixel[j][i], local_background_errors_per_pixel[j][i],
                                            flags[j]])
                
                # load light curve from file
                df = pd.read_csv(self.out_directory + f"annulus_light_curves/{fltr}_source_{i + 1}.csv")
                
                aligned_mask = df["quality_flag"] == "A"
                
                # reformat MJD to seconds from first observation
                df["MJD"] -= self.t_ref
                df["MJD"] *= 86400
                
                fig, ax = plt.subplots(nrows=3, tight_layout=True, figsize=(6.4, 2*4.8), sharex=True, gridspec_kw={"hspace": 0})
                
                ax[0].errorbar(df["MJD"].values[aligned_mask], df["flux"].values[aligned_mask], yerr=df["flux_error"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                ax[0].errorbar(df["MJD"].values[~aligned_mask], df["flux"].values[~aligned_mask], yerr=df["flux_error"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
                ax[0].set_ylabel("Flux [counts]")
                ax[0].set_title(f"{fltr} Source {i + 1}")
                
                ax[1].errorbar(df["MJD"].values[aligned_mask], df["local_background_per_pixel"].values[aligned_mask], yerr=df["local_background_error_per_pixel"].values[aligned_mask], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                ax[1].errorbar(df["MJD"].values[~aligned_mask], df["local_background_per_pixel"].values[~aligned_mask], yerr=df["local_background_error_per_pixel"].values[~aligned_mask], fmt="r.", ms=2, elinewidth=1, alpha=.2)
                ax[1].set_ylabel("Local background [cts/pixel]")
                
                ax[2].plot(df["MJD"].values[aligned_mask], df["flux"].values[aligned_mask]/df["local_background"].values[aligned_mask], "k.", ms=2)
                ax[2].plot(df["MJD"].values[~aligned_mask], df["flux"].values[~aligned_mask]/df["local_background"].values[~aligned_mask], "r.", ms=2, alpha=.2)
                ax[2].set_ylabel("SNR")
                ax[2].set_xlabel(f"Time from MJD {self.t_ref:.4f} [s]")
                
                for _ in ax:
                    _.minorticks_on()
                    _.tick_params(which="both", direction="in", top=True, right=True)
                
                # save light curve plot to file
                fig.savefig(self.out_directory + f"annulus_light_curves/{fltr}_source_{i + 1}.png")
                
                plt.close(fig)
    
    def __extract_aperture_and_annulus_light_curves(self, file, fltr, radius):
        
        fluxes, flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        local_backgrounds_per_pixel, local_background_errors_per_pixel = [], []
        
        if file != self.camera_files[f"{fltr}"][0]:
            try:
                transform = self.transforms[file]
            except:
                pass
        
        data = get_data(self.data_directory + file)
        error = np.sqrt(data*self.gains[file])  # Poisson noise
        
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
            aperture = CircularAperture(position, r=radius)
            annulus_aperture = CircularAnnulus(position, r_in=self.r_in_scale*radius, r_out=self.r_out_scale*radius)
            aperstats = ApertureStats(data, annulus_aperture, error=error, sigma_clip=SigmaClip(sigma=3, maxiters=10))
            aperture_area = aperture.area_overlap(data)
            
            # calculate local background per pixel
            if self.local_background_method == "median":
                local_background_per_pixel = aperstats.median
                local_background_error_per_pixel = aperstats.mad_std
            elif self.local_background_method == "mean":
                local_background_per_pixel = aperstats.mean
                local_background_error_per_pixel = aperstats.std
            
            # calculate total background in aperture
            total_bkg = local_background_per_pixel*aperture_area
            total_bkg_error = local_background_error_per_pixel*np.sqrt(aperture_area)
            
            phot_table = aperture_photometry(data, aperture, error=error)
            
            fluxes.append(phot_table["aperture_sum"].value[0])
            flux_errors.append(phot_table["aperture_sum_err"].value[0])
            local_backgrounds.append(total_bkg)
            local_background_errors.append(total_bkg_error)
            local_backgrounds_per_pixel.append(local_background_per_pixel)
            local_background_errors_per_pixel.append(local_background_error_per_pixel)
        
        return fluxes, flux_errors, local_backgrounds, local_background_errors, local_backgrounds_per_pixel, local_background_errors_per_pixel
    
    def photometry(self, phot_type: Literal["normal", "optimal"], tolerance: float = 5.):
        """
        Perform photometry on the images to extract source fluxes.

        Parameters
        ----------
        phot_type : Literal["normal", "optimal"]
            The type of photometry to perform. "normal" will extract fluxes using simple aperture photometry, while
            "optimal" will extract fluxes using the optimal photometry method outlined in Naylor 1998, MNRAS, 296, 339.
        tolerance : float, optional
            The tolerance for source position matching in standard deviations, by default 5. This parameter defines how
            far from the transformed catalog position a source can be while still being considered the same source. If
            the source is further than this tolerance, it will be considered a different source. If the alignments are
            good and the field is crowded, consider reducing this value. For poor alignments and/or uncrowded fields, 
            this value can be increased.
        """
        
        # determine which photometry function to use
        if phot_type == "normal":
            phot_function = self._extract_normal_light_curve
        elif phot_type == "annulus":
            phot_function = self._extract_optimal_light_curve
        else:
            raise ValueError(f"[OPTICAM] Photometry type {phot_type} not recognised.")
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"{phot_type}_light_curves"):
            os.mkdir(self.out_directory + f"{phot_type}_light_curves")
        
        print(f"[OPTICAM] Extracting {phot_type} fluxes ...")
        
        # for each camera
        for fltr in self.filters:
            
            # skip cameras with no images
            if len(self.camera_files[f"{fltr}"]) == 0:
                continue
            
            semimajor_sigma = np.median(self.catalogs[f"{fltr}"]["semimajor_sigma"].value)
            semiminor_sigma = np.median(self.catalogs[f"{fltr}"]["semiminor_sigma"].value)
            theta = np.median(self.catalogs[f"{fltr}"]["orientation"].value)
            
            observations = []
            
            for source in range(len(self.catalogs[f"{fltr}"])):
                print(f"[OPTICAM] Extracting {fltr} fluxes for source {source + 1} ...")
                
                with Pool(self.number_of_processors) as pool:
                    results = list(tqdm(pool.imap(partial(phot_function, source=source, fltr=fltr, 
                                                          tolerance=tolerance),
                                                  self.camera_files[f"{fltr}"]),
                                        total=len(self.camera_files[f"{fltr}"])))
                results = [result for result in results if result is not None]  # remove None values
                observations.append(len(results))
                
                try:
                    times, fluxes, flux_errors = zip(*results)
                except:
                    continue
                
                # save source light curve to file
                with open(self.out_directory + f"{phot_type}_light_curves/{fltr}_source_{source + 1}.csv", "w") as file:
                    
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(["MJD", "flux", "flux_error"])  # header
                    
                    # for each image
                    for i in range(len(fluxes)):
                        csvwriter.writerow([times[i], fluxes[i], flux_errors[i]])
                
                # reformat MJD to seconds from first observation
                ref = np.min(times)
                plot_times = np.array(times) - ref
                plot_times *= 86400
                
                fig, ax = plt.subplots(tight_layout=True)
                
                # plot light curve
                ax.errorbar(plot_times, fluxes, yerr=flux_errors, fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                
                ax.set_xlabel(f"Time from MJD {ref:.4f} [s]")
                ax.set_ylabel("Flux [counts]")
                ax.set_title(f"{fltr} Source {i + 1}")
                
                ax.minorticks_on()
                ax.tick_params(which="both", direction="in", top=True, right=True)
                
                # save light curve plot to file
                fig.savefig(self.out_directory + f"{phot_type}_light_curves/{fltr}_source_{i + 1}.png")
            
                plt.close(fig)
                
                
            fig, ax = plt.subplots(tight_layout=True)
            
            ax.bar(np.arange(len(self.catalogs[f"{fltr}"])) + 1, observations)
            
            ax.set_xlabel("Source")
            ax.set_ylabel("Number of epochs")
            
            fig.savefig(self.out_directory + f"diag/{phot_type}_{fltr}_observations.png")
            plt.close(fig)
    
    def _extract_normal_light_curve(self, file, source, fltr, tolerance):
        
        if file != self.camera_files[f"{fltr}"][0]:
            try:
                transform = self.transforms[file]
            except:
                return None
        
        data = get_data(self.data_directory + file)
        error = np.sqrt(data)  # Poisson noise
        bkg = self.background(data)
        clean_data = data - bkg.background
            
        catalog_position = (self.catalogs[f"{fltr}"]["xcentroid"][source], self.catalogs[f"{fltr}"]["ycentroid"][source])
        
        if file == self.camera_files[f"{fltr}"][0]:
            initial_position = catalog_position
        else:
            initial_position = astroalign.matrix_transform(catalog_position, transform)[0]
        
        segment_map = self.finder(clean_data, 5*bkg.background_rms)
        file_cat = SourceCatalog(clean_data, segment_map, background=bkg.background)
        file_tbl = file_cat.to_table()
        
        positions = np.array([[file_tbl["xcentroid"][i], file_tbl["ycentroid"][i]] for i in range(len(file_tbl))])
        distances = np.sqrt((positions[:, 0] - initial_position[0])**2 + (positions[:, 1] - initial_position[1])**2)
        
        if np.min(distances) > tolerance*np.sqrt(self.catalogs[f"{fltr}"]["semimajor_sigma"][source].value**2 + self.catalogs[f"{fltr}"]["semiminor_sigma"][source].value**2):
            return None
        
        i = np.argmin(distances)
        position = positions[i]
        
        semimajor_sigma = file_tbl["semimajor_sigma"][i].value
        semiminor_sigma = file_tbl["semiminor_sigma"][i].value
        theta = file_tbl["orientation"][i].value
        
        # define aperture
        aperture = EllipticalAperture(position, a=semimajor_sigma, b=semiminor_sigma, theta=theta)
        phot_table = aperture_photometry(clean_data, aperture, error=error)
        flux = phot_table["aperture_sum"].value[0]
        flux_error = phot_table["aperture_sum_err"].value[0]
        
        return self.times[file], flux, flux_error
    
    def _extract_optimal_light_curve(self, file, source, fltr, semimajor_sigma, semiminor_sigma, tolerance):
        
        print("[OPTICAM] optimal photometry is not yet implemented.")

    def _default_aperture_selector(self, aperture_sizes: ArrayLike) -> float:
        """
        Select the default aperture size.
        
        Parameters
        ----------
        aperture_sizes : ArrayLike
            The aperture sizes for the sources.
        
        Returns
        -------
        float
            The aperture size.
        """
        
        max_aperture = np.max(aperture_sizes)
        median_aperture = np.median(aperture_sizes)
        aperture_std = np.std(aperture_sizes)
        
        # if the maximum aperture is significantly larger than the median aperture, use the median aperture
        if max_aperture > median_aperture + aperture_std:
            return median_aperture
        # otherwise, use the maximum aperture
        else:
            return max_aperture


