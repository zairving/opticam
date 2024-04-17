from tqdm import tqdm
from astropy.table import QTable
import json
import astroalign
import numpy as np
from astropy.time import Time
import os
from astropy.io import fits
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
import pandas as pd
import csv

from opticam_new.helpers import rename_directory, get_data
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
        scale: float = 5,
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
        
        # format data directory
        self.data_directory = data_directory
        if self.data_directory[-1] != "/":
            self.data_directory += "/"
        
        rename_directory(self.data_directory)  # rename files to include camera number
        
        self.file_names = os.listdir(self.data_directory)  # get list of file names
        
        print("[OPTICAM] Scanning files ...")
        
        # check binning and parse observation times
        self.times = {}  # file name : file MJD
        binning = []
        for file_name in tqdm(self.file_names):
            with fits.open(self.data_directory + file_name) as hdul:
                # get file binning
                if hdul[0].header["BINNING"] not in binning:
                    binning.append(hdul[0].header["BINNING"])
                
                # parse observation time
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
                
                self.times.update({file_name: mjd})
        
        print("[OPTICAM] Done.")
        
        # if binning is not consistent, raise error
        if len(binning) > 1:
            raise ValueError("[OPTICAM] All images must have the same binning.")
        else:
            binning = binning[0]
        
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
        
        # set parameters
        self.scale = scale
        self.threshold = threshold
        
        # set parameters
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        # define colours for circling sources in catalogs 
        self.colours = list(mcolors.TABLEAU_COLORS.keys())
        self.colours.pop(self.colours.index("tab:brown"))  # remove this colour because it doesn't stand out on a dark background
        self.colours.pop(self.colours.index("tab:gray"))  # remove this colour because it doesn't stand out on a dark background
        
        # separate files by camera
        self.camera_files = {}
        for camera in [1, 2, 3]:
            self.camera_files[f"camera {camera}"] = sorted([file_name for file_name in self.file_names if f"C{camera}" in file_name])
        
        # write input parameters to file
        param_dict = {
            "threshold": threshold,
            # "background estimator": background_estimator.__class__.__name__,
            # "background RMS estimator": background_rms_estimator.__class__.__name__,
            "scale": scale,
            "number of files": len(self.file_names),
            "number of files for camera 1": len(self.camera_files["camera 1"]),
            "number of files for camera 2": len(self.camera_files["camera 2"]),
            "number of files for camera 3": len(self.camera_files["camera 3"]),
        }
        with open(self.out_directory + "misc/input.json", "w") as file:
            json.dump(param_dict, file, indent=4)
        
        # define background calculator
        if background is None:
            box_size = 16 if binning == "4x4" else 32 if binning == "2x2" else 64  # set box_size based on binning
            self.background = Background(box_size=box_size)
        else:
            self.background = background
        
        # define source finder
        if finder is None:
            npixels = 50 if binning == "4x4" else 100 if binning == "2x2" else 200  # set npixels based on binning
            border_width = 16 if binning == "4x4" else 32 if binning == "2x2" else 64  # set border_width based on binning
            self.finder = Finder(npixels=npixels, border_width=border_width)
        else:
            self.finder = finder
        
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
        for camera in [1, 2, 3]:
            try:
                self.catalogs.update({f"camera {camera}": QTable.read(self.out_directory + f"cat/camera_{camera}_catalog.ecsv", format="ascii.ecsv")})
                print("[OPTICAM] Read catalog for camera {camera} from file.")
                continue
            except:
                pass
    
    def initialise_catalogs(self) -> None:
        """
        Initialise the source catalogs for each camera.
        """
        
        print("[OPTICAM] Initialising catalogs ...")
        
        cat_fig, cat_ax = plt.subplots(ncols=3, tight_layout=True, figsize=(15, 5))
        
        # for each camera
        for camera in [1, 2, 3]:
            
            # if no images found for camera, skip
            if len(self.camera_files[f"camera {camera}"]) == 0:
                continue
            
            print(f"[OPTICAM] Aligning images for camera {camera} ...")
            
            reference_image = get_data(self.data_directory + self.camera_files[f"camera {camera}"][0])  # get reference image
            
            stacked_images = reference_image.copy()  # initialise stacked images as reference image
            
            cat_ax[camera - 1].set_title(f"Camera {camera}")
            cat_ax[camera - 1].set_xlabel("X")
            cat_ax[camera - 1].set_ylabel("Y")
            
            # align images
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(self._align_image, reference_image=reference_image),
                                              self.camera_files[f"camera {camera}"][1:]),
                                    total=len(self.camera_files[f"camera {camera}"]) - 1))
            
            results = [result for result in results if result is not None]  # remove None values
            
            transforms = {}  # define transforms as empty dictionary
            
            # update transforms dictionary
            for result in results:
                transforms.update(result)
            
            self.transforms.update(transforms)
            
            unaligned_files = []
            
            # define unaligned files list
            for file in self.camera_files[f"camera {camera}"]:
                if file not in self.transforms.keys() and file != self.camera_files[f"camera {camera}"][0]:
                    unaligned_files.append(file)
            
            self.unaligned_files += unaligned_files
            
            print(f"[OPTICAM] Done. {len(unaligned_files)} image(s) could not be aligned.")
            print(f"[OPTICAM] Stacking aligned images ...")
            
            # stack images
            # should be done one at a time rather than in parallel because of memory constraints
            for (file, transform) in tqdm(transforms.items()):
                data = get_data(self.data_directory + file)
                stacked_images += astroalign.apply_transform(SimilarityTransform(transform), data, reference_image)[0]
            
            print(f"[OPTICAM] Done.")
            
            # remove background from stacked images
            stacked_bkg = self.background(stacked_images)
            stacked_images -= stacked_bkg.background
            stacked_images = np.clip(stacked_images, 0, None)  # clip negative values to zero
            
            fig, ax = plt.subplots(tight_layout=True)
            
            ax.imshow(stacked_images, origin="lower", cmap="Greys_r", interpolation="nearest",
                      norm=simple_norm(stacked_images, stretch="log"))

            stacked_bkg.plot_meshes(ax=ax, outlines=True, marker='.', color='cyan', alpha=0.3)
            
            ax.set_title(f"Camera {camera}")
            
            fig.savefig(self.out_directory + f"diag/camera_{camera}_background_mesh.png")
            
            plt.close(fig)
            
            # identify sources in stacked images
            segment_map = self.finder(stacked_images, self.threshold*stacked_bkg.background_rms)
            
            # create catalog of sources in stacked images
            self.catalogs.update({f"camera {camera}": SourceCatalog(stacked_images, segment_map, background=stacked_bkg.background).to_table()})
            # write catalog to file
            self.catalogs[f"camera {camera}"].write(self.out_directory + f"cat/camera_{camera}_catalog.ecsv", format="ascii.ecsv",
                                            overwrite=True)
            
            # plot stacked images
            cat_ax[camera - 1].imshow(stacked_images, origin="lower", cmap="Greys_r", interpolation="nearest",
                                      norm=simple_norm(stacked_images, stretch="log"))

            radius = self.scale*np.max(self.catalogs[f"camera {camera}"]["semimajor_sigma"].value)
            
            # circle and label sources in stacked image            
            for i in range(len(self.catalogs[f"camera {camera}"])):
                cat_ax[camera - 1].add_patch(Circle(xy=(self.catalogs[f"camera {camera}"]["xcentroid"][i],
                                                    self.catalogs[f"camera {camera}"]["ycentroid"][i]),
                                                    radius=radius,
                                                    edgecolor=self.colours[i % len(self.colours)],
                                                    facecolor="none",
                                                    lw=1))
                ax[camera - 1].add_patch(Circle(xy=(self.catalogs[f"camera {camera}"]["xcentroid"][i],
                                                    self.catalogs[f"camera {camera}"]["ycentroid"][i]),
                                                    radius=2*radius,
                                                    edgecolor=self.colours[i % len(self.colours)],
                                                    facecolor="none",
                                                    lw=1, ls=":"))
                cat_ax[camera - 1].text(self.catalogs[f"camera {camera}"]["xcentroid"][i] + 1.05*radius,
                                    self.catalogs[f"camera {camera}"]["ycentroid"][i] + 1.05*radius, i + 1,
                                    color=self.colours[i % len(self.colours)])
        
        cat_fig.savefig(self.out_directory + "cat/catalogs.png")
        
        # save transforms to file
        with open(self.out_directory + "cat/transforms.json", "w") as file:
            json.dump(self.transforms, file, indent=4)
        
        if self.show_plots:
            plt.show(cat_fig)
        else:
            plt.close(cat_fig)
    
    def _align_image(self, file: str, reference_image: np.array) -> Dict[str, List]:
        """
        Align an image with respect to the reference image and return the transform.

        Parameters
        ----------
        file : str
            The name of the image file.
        reference_image : np.array
            The reference image used for alignment.

        Returns
        -------
        Dict[str, List]
            The file and its transformation parameters.
        """
        
        data = get_data(self.data_directory + file)
        
        try:
            # align image w.r.t reference image
            transform = astroalign.find_transform(reference_image, data)[0]
        except:
            return None
        
        # return transform and aligned image
        return {file: transform.params.tolist()}

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
        for camera in [1, 2, 3]:
            
            # skip cameras with no images
            if len(self.camera_files[f"camera {camera}"]) == 0:
                continue
            
            print(f"[OPTICAM] Creating frames for camera {camera} ...")
            
            # for each file
            for file in tqdm(self.camera_files[f"camera {camera}"]):
                
                # if file has no transform and is not the reference image, skip
                if file not in list(self.transforms.keys()) and file != self.camera_files[f"camera {camera}"][0]:
                    continue
                
                data = get_data(self.data_directory + file)
                bkg = self.background(data)
                clean_data = data - bkg.background
                clean_data = np.clip(clean_data, 0, None)  # clip negative values to zero
                
                # create GIF frame
                self._create_gif_frame(clean_data, file, camera)
            
            print(f"[OPTICAM] Done.")
            print(f"[OPTICAM] Compiling frames ...")
            
            # save GIF
            self._create_gif(camera, keep_frames)
            
            print(f"[OPTICAM] Done.")
    
    def _create_gif(self, camera: int, keep_frames: bool) -> None:
        """
        Create a gif from the frames saved in out_directory.

        Parameters
        ----------
        camera : int
            The camera number (1, 2, or 3).
        keep_frames : bool
            Whether to keep the frames after the gif is saved.
        """
        
        # load frames
        frames = []
        for file in self.camera_files[f"camera {camera}"]:
            try:
                frames.append(Image.open(self.out_directory + f"diag/camera_{camera}_gif_frames/{file.split(".")[0]}.png"))
            except:
                pass
        
        # save gif
        frames[0].save(self.out_directory + f"cat/camera_{camera}_images.gif", format="GIF", append_images=frames[1:], 
                       save_all=True, duration=200, loop=0)
        
        del frames  # delete frames after gif is saved to clear memory
        
        # delete frames after gif is saved
        if not keep_frames:
            for file in self.camera_files[f"camera {camera}"]:
                try:
                    os.remove(self.out_directory + f"{file.split(".")[0]}.png")
                except:
                    pass
    
    def _create_gif_frame(self, image: np.array, file: str, camera: int) -> None:
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
        
        fig, ax = plt.subplots(num=999, clear=True, tight_layout=True)  # set figure number to 999 to avoid conflict with other figures
        
        ax.imshow(image, origin="lower", cmap="Greys_r", interpolation="nearest",
                  norm=simple_norm(image, stretch="log"))
        
        # for each source
        for i in range(len(self.catalogs[f"camera {camera}"])):
            
            source_position = (self.catalogs[f"camera {camera}"]["xcentroid"][i], self.catalogs[f"camera {camera}"]["ycentroid"][i])
            
            try:
                aperture_position = astroalign.matrix_transform(source_position, self.transforms[file])[0]
            except:
                aperture_position = source_position
            
            radius = self.scale*self.catalogs[f"camera {camera}"]["semimajor_sigma"][i].value
            
            ax.add_patch(Circle(xy=(aperture_position), radius=radius,
                                    edgecolor=self.colours[i % len(self.colours)], facecolor="none", lw=1))
            ax[camera - 1].add_patch(Circle(xy=(self.catalogs[f"camera {camera}"]["xcentroid"][i],
                                                self.catalogs[f"camera {camera}"]["ycentroid"][i]),
                                            radius=2*radius, edgecolor=self.colours[i % len(self.colours)],
                                            facecolor="none", lw=1, ls=":"))
            ax.text(aperture_position[0] + 1.05*radius, aperture_position[1] + 1.05*radius, i + 1,
                        color=self.colours[i % len(self.colours)])
        
        ax.set_title(f"{file}")
        
        if not os.path.isdir(self.out_directory + f"diag/camera_{camera}_gif_frames"):
            os.mkdir(self.out_directory + f"diag/camera_{camera}_gif_frames")
        
        fig.savefig(self.out_directory + f"diag/camera_{camera}_gif_frames/{file.split(".")[0]}.png")
        
        fig.clear()
        plt.close(fig)
    
    def get_background(self):
        """
        Get the time-varying median background for each camera.
        """
        
        print("[OPTICAM] Getting background ...")
        
        fig, ax = plt.subplots(nrows=2, ncols=3, tight_layout=True, sharex=True, figsize=(2*6.4, 2*4.8))
        
        # for each camera
        for camera in [1, 2, 3]:
            
            # skip cameras with no images
            if len(self.camera_files[f"camera {camera}"]) == 0:
                continue
            
            print(f"[OPTICAM] Processing Camera {camera} files ...")
            
            times = np.array([self.times[file] for file in self.camera_files[f"camera {camera}"]])
            plot_times = (times - times.min())*86400  # convert to seconds from first observation
            
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(self._get_background_from_file, self.camera_files[f"camera {camera}"]),
                                    total=len(self.camera_files[f"camera {camera}"])))
            rms_values, median_values = zip(*results)
            
            # plot background
            ax[0, camera - 1].set_title(f"Camera {camera}")
            ax[0, camera - 1].plot(plot_times, rms_values, "k.", ms=2)
            ax[1, camera - 1].plot(plot_times, median_values, "k.", ms=2)
            
            # write background to file
            with open(self.out_directory + f"diag/camera_{camera}_background.txt", "w") as file:
                file.write("# MJD RMS median\n")
                for i in range(len(self.camera_files[f"camera {camera}"])):
                    file.write(f"{times[i]} {rms_values[i]} {median_values[i]}\n")

        ax[0, 0].set_ylabel("Median background RMS")
        ax[1, 0].set_ylabel("Median background")
        ax[1, 1].set_xlabel(f"Time from MJD {times.min():.4f} [s]")
        
        # save plot
        fig.savefig(self.out_directory + "diag/background.png")

        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
    
    def _get_background_from_file(self, file: str) -> Tuple[float, float]:
        """
        Get the median background RMS and median background from a file.

        Parameters
        ----------
        file : str
            The name of the file.

        Returns
        -------
        Tuple[float, float]
            The median background RMS and median background.
        """
        
        bkg = self.background(self.data_directory + file)
        
        return bkg.background_rms_median, bkg.background_median

    def forced_photometry(self, phot_type: Literal["aperture", "annulus"]) -> None:
        """
        Perform forced photometry on the images to extract source fluxes.

        Parameters
        ----------
        phot_type : Literal["aperture", "annulus"]
            The type of photometry to perform. "aperture" will extract fluxes using circular apertures, while "annulus"
            will extract fluxes using circular apertures and annuli (i.e., source flux - local background).

        Raises
        ------
        ValueError
            If the phot_type is not recognised.
        """
        
        # determine which photometry function to use
        if phot_type == "aperture":
            phot_function = self._extract_aperture_light_curves
        elif phot_type == "annulus":
            phot_function = self._extract_annulus_light_curves
        else:
            raise ValueError(f"[OPTICAM] Photometry type {phot_type} not recognised.")
        
        # create output directory if it does not exist
        if not os.path.isdir(self.out_directory + f"{phot_type}_light_curves"):
            os.mkdir(self.out_directory + f"{phot_type}_light_curves")
        
        print(f"[OPTICAM] Extracting {phot_type} fluxes ...")
        
        # for each camera
        for camera in [1, 2, 3]:
            
            # skip cameras with no images
            if len(self.camera_files[f"camera {camera}"]) == 0:
                continue
            
            # get aperture radius
            radius = self.scale*np.max(self.catalogs[f"camera {camera}"]["semimajor_sigma"].value)
            
            print(f"[OPTICAM] Processing Camera {camera} files ...")
            
            # extract fluxes
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(partial(phot_function, camera=camera, radius=radius),
                                              self.camera_files[f"camera {camera}"]),
                                    total=len(self.camera_files[f"camera {camera}"])))
            
            results = [result for result in results if result is not None]  # remove None values

            if phot_type == "annulus":
                times, fluxes, flux_errors, local_backgrounds, local_background_errors = zip(*results)  # unpack results
            else:
                times, fluxes, flux_errors = zip(*results)  # unpack results
            
            # for each source
            for i in range(len(fluxes[0])):
                # save source light curve to file
                with open(self.out_directory + f"{phot_type}_light_curves/camera_{camera}_source_{i + 1}.csv", "w") as file:
                    
                    csvwriter = csv.writer(file)
                    
                    if phot_type == "annulus":
                        csvwriter.writerow(["MJD", "flux", "flux_error", "local_background", "local_background_error"])
                    else:
                        csvwriter.writerow(["MJD", "flux", "flux_error"])
                    
                    # for each image
                    for j in range(len(fluxes)):
                        if phot_type == "annulus":
                            csvwriter.writerow([times[j], fluxes[j][i], flux_errors[j][i], local_backgrounds[j][i], local_background_errors[j][i]])
                        else:
                            csvwriter.writerow([times[j], fluxes[j][i], flux_errors[j][i]])
                
                # load light curve from file
                df = pd.read_csv(self.out_directory + f"{phot_type}_light_curves/camera_{camera}_source_{i + 1}.csv")
                
                # reformat MJD to seconds from first observation
                ref = df["MJD"].min()
                df["MJD"] -= ref
                df["MJD"] *= 86400
                
                if phot_type == "annulus":
                    
                    fig, ax = plt.subplots(nrows=3, tight_layout=True, figsize=(6.4, 2*4.8), sharex=True, gridspec_kw={"hspace": 0})
                    
                    ax[0].errorbar(df["MJD"], df["flux"], yerr=df["flux_error"], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                    ax[0].set_ylabel("Flux [counts]")
                    ax[0].set_title(f"Camera {camera}, Source {i + 1}")
                    
                    ax[1].errorbar(df["MJD"], df["local_background"], yerr=df["local_background_error"], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                    ax[1].set_ylabel("Local background [cts]")
                    
                    ax[2].plot(df["MJD"], df["flux"]/df["local_background"], "k-", lw=1)
                    ax[2].set_ylabel("SNR")
                    ax[2].set_xlabel(f"Time from MJD {ref:.4f} [s]")
                    
                    for _ in ax:
                        _.minorticks_on()
                        _.tick_params(which="both", direction="in", top=True, right=True)
                else:
                    fig, ax = plt.subplots(tight_layout=True)
                    
                    # plot light curve
                    ax.errorbar(df["MJD"], df["flux"], yerr=df["flux_error"], fmt="k.", ms=2, ecolor="grey", elinewidth=1)
                    
                    ax.set_xlabel(f"Time from MJD {ref:.4f} [s]")
                    ax.set_ylabel("Flux [counts]")
                    ax.set_title(f"Camera {camera}, Source {i + 1}")
                    
                    ax.minorticks_on()
                    ax.tick_params(which="both", direction="in", top=True, right=True)
                
                # save light curve plot to file
                fig.savefig(self.out_directory + f"{phot_type}_light_curves/camera_{camera}_source_{i + 1}.png")
                
                plt.close(fig)

    def _extract_aperture_light_curves(self, file, camera, radius):
        
        fluxes, flux_errors = [], []
        
        if file != self.camera_files[f"camera {camera}"][0]:
            try:
                transform = self.transforms[file]
            except:
                return None
        
        data = get_data(self.data_directory + file)
        error = np.sqrt(data)  # Poisson noise
        bkg = self.background(data)
        clean_data = data - bkg.background
        clean_data = np.clip(clean_data, 0, None)  # clip negative values to zero
        
        # for each source
        for i in range(len(self.catalogs[f"camera {camera}"])):
            
            catalog_position = (self.catalogs[f"camera {camera}"]["xcentroid"][i], self.catalogs[f"camera {camera}"]["ycentroid"][i])
            
            if file == self.camera_files[f"camera {camera}"][0]:
                position = catalog_position
            else:
                position = astroalign.matrix_transform(catalog_position, transform)[0]
            
            # define aperture
            aperture = CircularAperture(position, r=radius)
            phot_table = aperture_photometry(clean_data, aperture, error=error)
            fluxes.append(phot_table["aperture_sum"].value[0])
            flux_errors.append(phot_table["aperture_sum_err"].value[0])
        
        return self.times[file], fluxes, flux_errors

    def _extract_annulus_light_curves(self, file, camera, radius):
        
        fluxes, flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        
        if file != self.camera_files[f"camera {camera}"][0]:
            try:
                transform = self.transforms[file]
            except:
                return None
        
        data = get_data(self.data_directory + file)
        error = np.sqrt(data)  # Poisson noise
        bkg = self.background(data)
        clean_data = data - bkg.background
        clean_data = np.clip(clean_data, 0, None)  # clip negative values to zero
        
        # for each source
        for i in range(len(self.catalogs[f"camera {camera}"])):
            
            catalog_position = (self.catalogs[f"camera {camera}"]["xcentroid"][i], self.catalogs[f"camera {camera}"]["ycentroid"][i])
            
            if file == self.camera_files[f"camera {camera}"][0]:
                position = catalog_position
            else:
                position = astroalign.matrix_transform(catalog_position, transform)[0]
            
            # define aperture
            aperture = CircularAperture(position, r=radius)
            annulus_aperture = CircularAnnulus(position, r_in=radius, r_out=2*radius)
            aperstats = ApertureStats(clean_data, annulus_aperture, error=error)
            aperture_area = aperture.area_overlap(clean_data)
            total_bkg = aperstats.mean*aperture_area # total background in annulus
            total_bkg_error = aperstats.std*np.sqrt(aperture_area) # error on total background
            phot_table = aperture_photometry(clean_data, aperture, error=error)
            
            fluxes.append(phot_table["aperture_sum"].value[0] - total_bkg)
            flux_errors.append(np.sqrt(phot_table["aperture_sum_err"].value[0]**2 + total_bkg_error**2))
            local_backgrounds.append(total_bkg)
            local_background_errors.append(total_bkg_error)
        
        return self.times[file], fluxes, flux_errors, local_backgrounds, local_background_errors


