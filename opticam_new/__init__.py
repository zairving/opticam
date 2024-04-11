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
from photutils.background import Background2D, SExtractorBackground, StdBackgroundRMS
from photutils.segmentation import SourceFinder, SourceCatalog
from photutils.aperture import ApertureStats, aperture_photometry, CircularAperture, CircularAnnulus
from skimage.transform import SimilarityTransform
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image
from typing import Union, Tuple, List, Dict, Literal, Callable, Type
from numpy.typing import ArrayLike
import pandas as pd
import csv


class Reducer:
    """
    Helper class for reducing OPTICAM data.
    """
    
    def __init__(
        self,
        data_directory: str,
        out_directory: str,
        sigma: int = 3,
        threshold: float = 5,
        background_estimator = SExtractorBackground(),
        background_rms_estimator = StdBackgroundRMS(),
        connectivity: Literal[4, 8] = 8,
        nlevels: int = 32,
        contrast: float = 1e-3,
        npixels: int = None,
        box_size: Union[int, ArrayLike] = None,
        border_width: int = None,
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
        sigma : int, optional
            The sigma value for sigma clipping, by default 3.
        threshold : float, optional
            The threshold value for source detection, by default 5. This value is multiplied by the background RMS to
            determine the threshold for source detection.
        background_estimator : Union[Type[BackgroundEstimator], Callable[[SigmaClip], ArrayLike]], optional
            The background estimator to use, by default SExtractorBackground. Any photutils background estimator
            can be used, as well as any custom function or callable object that takes a SigmaClip object as input and
            returns an ndarray.
        background_rms_estimator : Union[Type[BackgroundRMSEstimator], Callable[[SigmaClip], ArrayLike]], optional
            The background RMS estimator to use, by default StdBackgroundRMS. Any photutils background RMS estimator
            can be used, as well as any custom function or callable object that takes a SigmaClip object as input and
            returns an ndarray.
        connectivity : Literal[4, 8], optional
            The connectivity of the sources, by default 8. 8 means that source pixels can be connected diagonally as
            well as orthogonally. 4 means that source pixels can only be connected orthogonally.
        nlevels : int, optional
            The number of multi-thresholding levels to use for deblending sources, by default 32.
        contrast : float, optional
            The fraction of the total source flux that a local peak must have to be detected as a separate source, by
            default 1e-3.
        npixels : int, optional
            The number of connected pixels that constitutes a source, by default None. If None, the number of pixels
            will be set based on the binning of the images.
        box_size : Union[int, ArrayLike], optional
            The size of the box to use for background subtraction, by default None. If None, the box size will be set
            based on the binning of the images.
        border_width : int, optional
            Sources within this distance from the edge of the image will be ignored, by default None. If None, the border
            width will be set based on the binning of the images.
        scale : float, optional
            The scale factor for the source apertures, by default 5. Sources are modelled as bivariate Gaussians, so a
            scale factor of 5 means that the aperture will be 5 times the standard deviation of the Gaussian along its
            major axis.
        number_of_processors : int, optional
            The number of processors to use for parallel processing, by default cpu_count(). Increasing this number will
            increase performance at the cost of increased memory usage. To avoid prohibitive memory usage, some methods
            are not parallelised.
        show_plots : bool, optional
            Whether to show plots (for example, if being run in a notebook environment), by default True.
        """
        
        # format data directory
        self.data_directory = data_directory
        if self.data_directory[-1] != "/":
            self.data_directory += "/"
        
        rename_directory(self.data_directory)  # rename files to include camera number
        
        self.file_names = os.listdir(self.data_directory)  # get list of file names
        
        print("[OPTICAM] Scanning files ...")
        
        # check binning and whether GPSTIME is in file headers
        # early OPTICAM data require GPSTIME to be used as the time key, while later data use UT
        binning = []
        gpstime_in_header = False
        for file_name in tqdm(self.file_names):
            with fits.open(self.data_directory + file_name) as hdul:
                if hdul[0].header["BINNING"] not in binning:
                    binning.append(hdul[0].header["BINNING"].replace(" ", ""))
                if "GPSTIME" in hdul[0].header.keys():
                    gpstime_in_header = True
        
        print("[OPTICAM] Done.")
        
        # set date key so image times can be parsed correctly
        if gpstime_in_header:
            self.date_key = "GPSTIME"
        else:
            self.date_key = "UT"
        
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
        self.sigma_clip = SigmaClip(sigma=sigma)
        
        # set background estimator
        if callable(background_estimator):
            self.bkg_estimator = background_estimator
        else:
            raise ValueError(f"[OPTICAM] Background estimator {background_estimator} is not callable.")
        
        # set background RMS estimator
        if callable(background_rms_estimator):
            self.bkgrms_estimator = background_rms_estimator
        else:
            raise ValueError(f"[OPTICAM] Background RMS estimator {background_rms_estimator} is not callable.")
        
        # set parameters
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        # define colours for circling sources in catalogs 
        self.colours = list(mcolors.TABLEAU_COLORS.keys())
        self.colours.pop(self.colours.index("tab:brown"))  # remove this colour because it doesn't stand out on a dark background
        self.colours.pop(self.colours.index("tab:gray"))  # remove this colour because it doesn't stand out on a dark background
        
        # set parameters based on binning
        if binning == "1x1":
            self.border_width = border_width if border_width is not None else 64
            self.box_size = box_size if box_size is not None else 64
            npixels = npixels if npixels is not None else 200
        elif binning == "2x2":
            self.border_width = border_width if border_width is not None else 32
            self.box_size = box_size if box_size is not None else 32
            npixels = npixels if npixels is not None else 100
        elif binning == "4x4":
            self.border_width = border_width if border_width is not None else 16
            self.box_size = box_size if box_size is not None else 16
            npixels = npixels if npixels is not None else 50
        else:
            raise ValueError(f"[OPTICAM] Binning {binning} not recognised.")
        
        # separate files by camera
        self.camera_files = {}
        for camera in [1, 2, 3]:
            self.camera_files[f"camera {camera}"] = sorted([file_name for file_name in self.file_names if f"C{camera}" in file_name])
        
        # write input parameters to file
        param_dict = {
            "sigma": sigma,
            "threshold": threshold,
            "background estimator": background_estimator.__class__.__name__,
            "background RMS estimator": background_rms_estimator.__class__.__name__,
            "npixels": npixels,
            "connectivity": connectivity,
            "nlevels": nlevels,
            "contrast": contrast,
            "box_size": self.box_size,
            "border_width": self.border_width,
            "scale": scale,
            "number of files": len(self.file_names),
            "number of files for camera 1": len(self.camera_files["camera 1"]),
            "number of files for camera 2": len(self.camera_files["camera 2"]),
            "number of files for camera 3": len(self.camera_files["camera 3"]),
        }
        with open(self.out_directory + "misc/input.json", "w") as file:
            json.dump(param_dict, file, indent=4)
        
        # define source finder
        self.finder = SourceFinder(npixels=npixels, connectivity=connectivity, nlevels=nlevels, contrast=contrast, progress_bar=False)
    
    def initialise_catalogs(self, overwrite: bool = False) -> None:
        """
        Initialise the source catalogs for each camera.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite existing catalogs, by default False.
        """
        
        print("[OPTICAM] Initialising catalogs ...")
        
        self.transforms = {}  # define transforms as empty dictionary
        self.unaligned_files = []  # define unaligned files as empty list
        self.catalogs = {}  # define catalogs as empty dictionary
        
        cat_fig, cat_ax = plt.subplots(ncols=3, tight_layout=True, figsize=(15, 5))
        
        # load transforms from file
        if os.path.isfile(self.out_directory + "cat/transforms.json") and not overwrite:
            with open(self.out_directory + "cat/transforms.json", "r") as file:
                self.transforms.update(json.load(file))
            print("[OPTICAM] Read transforms from file.")
        
        # for each camera
        for camera in [1, 2, 3]:
            
            # if no images found for camera, skip
            if len(self.camera_files[f"camera {camera}"]) == 0:
                continue
            
            if os.path.isfile(self.out_directory + f"cat/camera_{camera}_catalog.ecsv") and not overwrite:
                try:
                    self.catalogs.update({f"camera {camera}": QTable.read(self.out_directory + f"cat/camera_{camera}_catalog.ecsv", format="ascii.ecsv")})
                    print("[OPTICAM] Read catalog for camera {camera} from file.")
                    continue
                except:
                    pass
            
            print(f"[OPTICAM] Aligning images for camera {camera} ...")
            
            # align all images w.r.t first image
            with fits.open(self.data_directory + self.camera_files[f"camera {camera}"][0]) as hdul:
                reference_image = np.array(hdul[0].data, dtype=np.float64)
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
            # must be done one at a time rather than in parallel because of memory constraints (can't hold thousands of images in memory at once)
            for (file, transform) in tqdm(transforms.items()):
                with fits.open(self.data_directory + file) as hdul:
                    data = np.array(hdul[0].data, dtype=np.float64)
                stacked_images += astroalign.apply_transform(SimilarityTransform(transform), data, reference_image)[0]
            
            print(f"[OPTICAM] Done.")
            
            # remove background from stacked images
            stacked_bkg = Background2D(stacked_images, self.box_size, sigma_clip=self.sigma_clip, bkg_estimator=self.bkg_estimator, bkgrms_estimator=self.bkgrms_estimator)
            stacked_images -= stacked_bkg.background
            
            fig, ax = plt.subplots(tight_layout=True)
            
            ax.imshow(stacked_images, origin="lower", cmap="Greys_r", interpolation="nearest",
                      norm=simple_norm(stacked_images, stretch="log"))

            stacked_bkg.plot_meshes(ax=ax, outlines=True, marker='.', color='cyan', alpha=0.3)
            
            ax.set_title(f"Camera {camera}")
            
            fig.savefig(self.out_directory + f"diag/camera_{camera}_background_mesh.png")
            
            plt.close(fig)
            
            # identify sources in stacked images
            segment_map = self.finder(stacked_images, self.threshold*stacked_bkg.background_rms)
            segment_map.remove_border_labels(border_width=self.border_width, relabel=True)  # remove sources 30 pixels from edge
            
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
        list
            The transformation parameters as a .
        """
        
        with fits.open(self.data_directory + file) as hdul:
            data = np.array(hdul[0].data, dtype=np.float64)

        bkg = Background2D(data, self.box_size, sigma_clip=self.sigma_clip, bkg_estimator=self.bkg_estimator, bkgrms_estimator=self.bkgrms_estimator)
        clean_data = data - bkg.background
        
        try:
            # align image w.r.t reference image
            transform = astroalign.find_transform(reference_image, clean_data)[0]
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
                
                clean_data = self._read_file(file)[1]  # get background subtracted image
                
                # if file has no transform and is not the reference image, skip
                if file not in list(self.transforms.keys()) and file != self.camera_files[f"camera {camera}"][0]:
                    continue
                
                # create GIF frame
                self._create_gif_frame(clean_data, file, camera)
            
            print(f"[OPTICAM] Done.")
            print(f"[OPTICAM] Compiling frames ...")
            
            # save GIF
            self._create_gif(camera, keep_frames)
            
            print(f"[OPTICAM] Done.")
    
    def _create_gif(self, camera: int, keep_frames: bool) -> None:
        """
        Create a gif from the frames saved in the out_directory.

        Parameters
        ----------
        camera : int
            The camera number (1, 2, or 3).
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
            
            ax.add_patch(Circle(xy=(aperture_position),
                                    radius=radius,
                                    edgecolor=self.colours[i % len(self.colours)],
                                    facecolor="none",
                                    lw=1),
                            )
            ax.text(aperture_position[0] + 1.05*radius, aperture_position[1] + 1.05*radius, i + 1,
                        color=self.colours[i % len(self.colours)])
        
        ax.set_title(f"{file}")
        
        if not os.path.isdir(self.out_directory + f"diag/camera_{camera}_gif_frames"):
            os.mkdir(self.out_directory + f"diag/camera_{camera}_gif_frames")
        
        fig.savefig(self.out_directory + f"diag/camera_{camera}_gif_frames/{file.split(".")[0]}.png")
        
        fig.clear()
        plt.close(fig)
    
    def _read_file(self, file: str) -> tuple[float, np.array, np.array]:
        
        with fits.open(self.data_directory + file) as hdul:
            data = hdul[0].data

        error = np.sqrt(data)  # assuming Poisson noise
        bkg = Background2D(data, self.box_size, sigma_clip=self.sigma_clip, bkg_estimator=self.bkg_estimator, bkgrms_estimator=self.bkgrms_estimator)
        clean_data = data - bkg.background
        clean_data = np.clip(clean_data, 0, None)  # remove negative values
        
        return self._parse_file_header_time(file), clean_data, error, bkg
    
    def _parse_file_header_time(self, file: str) -> float:
        
        with fits.open(self.data_directory + file) as hdul:
            if self.date_key == "UT":
                try:
                    time = Time(hdul[0].header["UT"].replace(" ", "T"), format="fits").mjd
                except:
                    raise KeyError(f"[OPTICAM] Could not find {self.date_key} key in header.")
            elif self.date_key == "GPSTIME":
                try:
                    temp = hdul[0].header["GPSTIME"]
                except:
                    raise KeyError(f"[OPTICAM] Could not find {self.date_key} key in header.")
                try:
                    split_temp = temp.split(" ")
                    date = split_temp[0]
                    gps_time = split_temp[1].split(".")[0]  # remove decimal seconds
                    time = Time(date + "T" + gps_time, format="fits").mjd
                except:
                    raise ValueError(f"[OPTICAM] Could not parse {self.date_key} key in header.")

        return time

    def get_header_times(self):
        
        print("[OPTICAM] Getting header times ...")
        
        # for each camera
        for camera in [1, 2, 3]:
            
            print(f"[OPTICAM] Processing Camera {camera} files ...")
            
            # get times from file headers
            times = []
            for file in tqdm(self.camera_files[f"camera {camera}"]):
                times.append(self._parse_file_header_time(file))
            
            # if camera has files
            if len(times) > 0:
            
                times = sorted(times)  # sort times (in case file names are not labelled in order)
                
                fig, ax = plt.subplots(ncols=2, tight_layout=True, figsize=(1.5*6.4, 4.8))
                
                dt = np.diff(times)*24*60*60  # convert difference in times from days to seconds
                bin_edges = np.arange(0, round(np.max(dt)) + 2, 1)  # define bin edges
                
                # plot time differences
                ax[0].step(np.arange(len(times) - 1), dt, where="mid", color="k", lw=1)
                ax[0].set_xlabel("Epoch/file number")
                ax[0].set_ylabel(r"$\Delta t_{\rm start} [s]$")
                ax[0].minorticks_on()
                
                # plot histogram of time differences (useful for identifying number of outliers)
                ax[1].hist(dt, bins=bin_edges, color="k", histtype="step", lw=1)
                ax[1].set_yscale("log")
                ax[1].set_xlabel(r"$\Delta t_{\rm start} [s]$")
                ax[1].minorticks_on()
                
                fig.savefig(self.out_directory + f"diag/camera_{camera}_header_times.png")
                
                if self.show_plots:
                    plt.show()
                else:
                    plt.close(fig)

    def get_background(self):
        """
        Get the time-varying background for each camera.
        """
        
        print("[OPTICAM] Getting background ...")
        
        fig, ax = plt.subplots(nrows=2, ncols=3, tight_layout=True, sharex=True, figsize=(2*6.4, 2*4.8))
        
        # for each camera
        for camera in [1, 2, 3]:
            
            # skip cameras with no images
            if len(self.camera_files[f"camera {camera}"]) == 0:
                continue
            
            print(f"[OPTICAM] Processing Camera {camera} files ...")
            
            # get background from files
            with Pool(self.number_of_processors) as pool:
                results = list(tqdm(pool.imap(self._get_background_from_file, self.camera_files[f"camera {camera}"]),
                                    total=len(self.camera_files[f"camera {camera}"])))
            times, rmss, medians = zip(*results)  # unpack results
            
            # plot background
            ax[0, camera - 1].set_title(f"Camera {camera}")
            ax[0, camera - 1].plot(times, rmss, "k.", ms=2)
            ax[1, camera - 1].plot(times, medians, "k.", ms=2)
            
            # write background to file
            with open(self.out_directory + f"diag/camera_{camera}_background.txt", "w") as file:
                file.write("# MJD RMS median\n")
                for i in range(len(times)):
                    file.write(f"{times[i]} {rmss[i]} {medians[i]}\n")

        ax[0, 0].set_ylabel("Median background RMS")
        ax[1, 0].set_ylabel("Median background")
        
        # save plot
        fig.savefig(self.out_directory + "diag/background.png")

        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def _get_background_from_file(self, file):
        
        time, clean_data, error, bkg = self._read_file(file)
        
        return time, bkg.background_rms_median, bkg.background_median
    
    def catalog_with_annuli(self):
        fig, ax = plt.subplots(ncols=3, tight_layout=True, figsize=(15, 5))
        
        print("[OPTICAM] Initialising catalogs ...")
        
        # for each camera
        for camera in [1, 2, 3]:
            
            # if no images found for camera, skip
            if len(self.camera_files[f"camera {camera}"]) == 0:
                continue
            
            print(f"[OPTICAM] Aligning images for camera {camera} ...")
            
            # open reference image
            with fits.open(self.data_directory + self.camera_files[f"camera {camera}"][0]) as hdul:
                data = hdul[0].data
            
            # subtract background from image
            bkg = Background2D(data, self.box_size, sigma_clip=self.sigma_clip, bkg_estimator=self.bkg_estimator, bkgrms_estimator=self.bkgrms_estimator)
            reference_image = data - bkg.background
            
            # plot image
            ax[camera - 1].imshow(reference_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                                norm=simple_norm(reference_image, stretch="log"))
            
            # identify sources in image
            segment_map = self.finder(reference_image, 5*bkg.background_rms)
            segment_map.remove_border_labels(border_width=self.border_width, relabel=True)  # remove sources 30 pixels from edge
            
            # circle and label sources in image            
            for i in range(len(self.catalogs[f"camera {camera}"])):
                radius = self.scale*self.catalogs[f"camera {camera}"]["semimajor_sigma"][i].value
                ax[camera - 1].add_patch(Circle(xy=(self.catalogs[f"camera {camera}"]["xcentroid"][i],
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
                ax[camera - 1].text(self.catalogs[f"camera {camera}"]["xcentroid"][i] + 1.05*radius,
                                    self.catalogs[f"camera {camera}"]["ycentroid"][i] + 1.05*radius, i + 1,
                                    color=self.colours[i % len(self.colours)])
            
            ax[camera - 1].set_title(f"Camera {camera}")
            ax[camera - 1].set_xlabel("X")
            ax[camera - 1].set_ylabel("Y")
        
        fig.savefig(self.out_directory + "catalogs_with_annuli.png")
        
        if self.show_plots:
            plt.show()
        else:
            fig.clear()
            plt.close(fig)

    def photometry(self, phot_type: Literal["aperture", "annulus"]) -> None:
        """
        Perform photometry on the images to extract source fluxes.

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
        
        time, clean_data, error, bkg = self._read_file(file)
        
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
        
        return time, fluxes, flux_errors

    def _extract_annulus_light_curves(self, file, camera, radius):
        
        fluxes, flux_errors = [], []
        local_backgrounds, local_background_errors = [], []
        
        if file != self.camera_files[f"camera {camera}"][0]:
            try:
                transform = self.transforms[file]
            except:
                return None
        
        time, clean_data, error, bkg = self._read_file(file)
        
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
        
        return time, fluxes, flux_errors, local_backgrounds, local_background_errors




class Photometer:
    
    def __init__(self, data_directory: str, out_directory: str, date_key: str = "UT",
                 number_of_processors: int = cpu_count(), show_plots: bool = True):
        
        self.data_directory = data_directory
        if self.data_directory[-1] != "/":
            self.data_directory += "/"
        
        if not os.path.isdir(self.data_directory):
            raise FileNotFoundError(f"[OPTICAM] {self.data_directory} not found.")
        
        self.out_directory = out_directory
        if self.out_directory[-1] != "/":
            self.out_directory += "/"
        
        if not os.path.isdir(self.out_directory):
            raise FileNotFoundError(f"[OPTICAM] {self.out_directory} not found. Make sure to run Reducer first so that out_directory contains the necessary files for photometry.")
        
        self.date_key = date_key
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
    
    def get_relative_flux(self, target: List[int], comparison: List[int], flux_type: Literal["aperture", "annulus"],
                          save: bool = True, label: str = None) -> Union[Tuple[ArrayLike, ArrayLike, ArrayLike], None]:
        """
        Compute the relative flux between a target and comparison source for each camera.

        Parameters
        ----------
        target : List[int]
            The catalog IDs of the target source (i.e., [camera 1 ID, camera 2 ID, camera 3 ID]).
        comparison : List[int]
            The catalog IDs of the comparison source (i.e., [camera 1 ID, camera 2 ID, camera 3 ID]).
        save : bool, optional
            Whether to save the resulting light curves to out_directory, by default True.
        label : str, optional
            The label to add to the file name is save is True, by default None.

        Returns
        -------
        Union[Tuple[ArrayLike, ArrayLike, ArrayLike], None]
            If save is False, the time, relative flux, and relative flux error arrays are returned. Otherwise,
            nothing is returned.
        """
        
        if save:
            assert label is not None, "A label must be provided if save is True."

        if flux_type == "aperture":
            save_label = "aperture_light_curve"
            light_curve_dir = "aperture_light_curves"
        elif flux_type == "annulus":
            save_label = "annulus_light_curve"
            light_curve_dir = "annulus_light_curves"
        else:
            print(f"[OPTICAM] Flux type {flux_type} is not supported.")
            return None
        
        t, f, ferr = [], [], []
        
        fig, ax = plt.subplots(nrows=3, tight_layout=True, sharex=True, figsize=(1.5*6.4, 2*4.8))
        
        # for each camera
        for camera in [1, 2, 3]:
            try:
                source_df = pd.read_csv(self.out_directory + f"{light_curve_dir}/camera_{camera}_source_{target[camera - 1]}.csv")
                comparison_df = pd.read_csv(self.out_directory + f"{light_curve_dir}/camera_{camera}_source_{comparison[camera - 1]}.csv")
            except:
                print(f"[OPTICAM] No raw flux files found for camera {camera}, skipping ...")
                print(self.out_directory + f"{light_curve_dir}/camera_{camera}_source_{target[camera - 1]}.csv")
                print(f"{light_curve_dir}/camera_{camera}_source_{comparison[camera - 1]}.csv")
                continue
            
            # ensure time arrays match, otherwise relative flux cannot be calculated
            if not np.array_equal(source_df["MJD"].values, comparison_df["MJD"].values):
                print(f"[OPTICAM] Time arrays for camera {camera} do not match, skipping ...")
                continue
            
            relative_flux = source_df["flux"].values/comparison_df["flux"].values
            relative_flux_error = relative_flux*np.sqrt(np.square(source_df["flux_error"].values/source_df["flux"].values) + np.square(comparison_df["flux_error"].values/comparison_df["flux"].values))
            
            t.append(source_df["MJD"].values)
            f.append(relative_flux)
            ferr.append(np.abs(relative_flux_error))
            
            ax[camera - 1].errorbar(source_df["MJD"].values, relative_flux, np.abs(relative_flux_error), fmt="k.", ms=2, ecolor="grey", elinewidth=1)
            ax[camera - 1].set_title(f"Camera {camera} (Source ID: {target[camera - 1]}, Comparison ID: {comparison[camera - 1]})")
        
        ax[2].set_xlabel("MJD")
        ax[1].set_ylabel("Relative flux")

        fig.savefig(self.out_directory + f"{label}_{save_label}.png")

        if self.show_plots:
            plt.show()
        
        if save:
            # for each camera
            for i in range(len(t)):
                # create light curve dictionary
                light_curve = {"MJD": t[i], "flux": f[i], "flux_error": ferr[i]}
                self._save_light_curve(light_curve, f"camera_{i + 1}_{label}_{save_label}.txt", target=target[i], comparison=comparison[i])
        else:
            return t, f, ferr
    
    def _save_light_curve(self, light_curve: dict, file_name: str, target: int, comparison: int) -> None:
        """
        Saves the light curve to out_directory under file_name.

        Parameters
        ----------
        light_curve : dict
            The light curve to be saved.
        file_name : str
            The name of the file.
        target : int
            The target source's catalog ID.
        comparison : int
            The comparison source's catalog ID.
        """
        
        header = list(light_curve.keys())

        with open(self.out_directory + file_name, "w") as file:
            file.write(f"# target: {target}\n")
            file.write(f"# comparison: {comparison}\n")
            
            for key in header:
                file.write(f"{key} ")
            file.write("\n")
            
            for i in range(len(light_curve[header[0]])):
                for key in header:
                    file.write(f"{light_curve[key][i]} ")
                file.write("\n")

    def get_relative_error_light_curve(self, target: str, flux_type_1: str, flux_type_2: str) -> None:
        
        fig, ax = plt.subplots(nrows=3, tight_layout=True, sharex=True, figsize=(6.4, 2*4.8))
        
        for camera in [1, 2, 3]:
            try:
                df1 = pd.read_csv(self.out_directory + f"camera_{camera}_{target}_{flux_type_1}_light_curve.txt", delimiter=" ", comment="#")
            except:
                print(f"[OPTICAM] {self.out_directory}camera_{camera}_{target}_{flux_type_1}_light_curve.txt could not be loaded, skipping ...")
                continue
            
            try:
                df2 = pd.read_csv(self.out_directory + f"camera_{camera}_{target}_{flux_type_2}_light_curve.txt", delimiter=" ", comment="#")
            except:
                print(f"[OPTICAM] {self.out_directory}camera_{camera}_{target}_{flux_type_2}_light_curve.txt could not be loaded, skipping ...")
                continue
            
            assert np.array_equal(df1["MJD"].values, df2["MJD"].values), f"Time arrays for camera {camera} do not match."
            
            f = (df1["flux"].values - df2["flux"].values)/df1["flux"].values
            temp = np.sqrt(np.square(df1["flux_error"].values/df1["flux"].values) + np.square(df2["flux_error"].values/df2["flux"].values))
            temp2 = np.sqrt(np.square(temp/(df1["flux"].values - df2["flux"].values)) + np.square(df1["flux_error"].values/df1["flux"].values))
            ferr = np.abs(f)*temp2
            
            ax[camera - 1].errorbar(df1["MJD"], f, ferr, fmt="k.", ms=2, ecolor="grey", elinewidth=1)
            
        ax[2].set_xlabel("MJD")
        ax[1].set_ylabel(r"$(F_{\rm " + flux_type_1.replace("_", "\\_") + r"} - F_{\rm " + flux_type_2.replace("_", "\\_") + r"})/F_{\rm " + flux_type_1.replace("_", "\\_") + r"}$")
        
        fig.savefig(self.out_directory + f"{target}_relative_error_light_curve.png")




def rename_directory(directory):
    
    for file_name in os.listdir(directory):
        
        name = file_name.split(".")[0]
        extension = "." + file_name.split(".")[1]
        
        try:
            extension += "." + file_name.split(".")[2]  # if file is gzipped (e.g., .fits.gz)
        except:
            pass
        
        if not ("fit" in extension or "fits" in extension):
            continue
        
        if "C1" in file_name or "C2" in file_name or "C3" in file_name:
            continue

        source = directory + file_name
        
        if 'u' in name or 'g' in name:
            ch = 'C1'
            if 'u' in name:
                key = 'u'
            else:
                key = 'g'
        if 'r' in name:
            ch = 'C2'
            key = 'r'
        if 'i' in name or 'z' in name:
            ch = 'C3'
            if 'i' in name:
                key = 'i'
            else:
                key = 'z'
        
        new_name = name.split(key)[0] + ch + key + name.split(key)[1] + extension
        destination = directory + new_name
    
        os.rename(source, destination)











