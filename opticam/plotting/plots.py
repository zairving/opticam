from typing import Dict, List

from astropy.table import QTable
from astropy.visualization import simple_norm
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import os.path
from pandas import DataFrame

from opticam.background.global_background import BaseBackground
from opticam.utils.constants import catalog_colors


def plot_catalogs(
    out_directory: str,
    stacked_images: Dict[str, NDArray],
    catalogs: Dict[str, QTable],
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the source catalogs.
    
    Parameters
    ----------
    out_directory : str
        The directory path to which the resulting plot will be saved.
    filters : List[str]
        The catalog filters.
    stacked_images : Dict[str, NDArray]
        The stacked images for each filter {filter: image}.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    fig, axes = plt.subplots(
        ncols=len(stacked_images),
        tight_layout=True,
        figsize=(
            len(stacked_images) * 5,
            5,
            ),
        )
    
    if len(stacked_images) == 1:
        axes = [axes]
    
    for i, fltr in enumerate(stacked_images):
        
        plot_image = np.clip(stacked_images[fltr], 0, None)  # clip negative values to zero for better visualisation
        
        # plot stacked image
        axes[i].imshow(
            plot_image,
            origin="lower",
            cmap="Greys_r",
            interpolation="nearest",
            norm=simple_norm(
                plot_image,
                stretch="log",
                ),
            )
        
        # get aperture radius
        radius = 5 * np.median(catalogs[fltr]["semimajor_sigma"].value)
        
        for j in range(len(catalogs[fltr])):
            # label sources
            axes[i].add_patch(
                Circle(
                    xy=(
                        catalogs[fltr]["xcentroid"][j],
                        catalogs[fltr]["ycentroid"][j],
                        ),
                    radius=radius,
                    edgecolor=catalog_colors[j % len(catalog_colors)],
                    facecolor="none",
                    lw=1,
                    ),
                )
            axes[i].text(
                catalogs[fltr]["xcentroid"][j] + 1.05 * radius,
                catalogs[fltr]["ycentroid"][j] + 1.05 * radius,
                j + 1,  # source number
                color=catalog_colors[j % len(catalog_colors)],
                )
            
            # label plot
            axes[i].set_title(fltr)
            axes[i].set_xlabel("X")
            axes[i].set_ylabel("Y")
    
    if save:
        fig.savefig(os.path.join(out_directory, "cat/catalogs.png"), dpi=200)
    
    if show:
        plt.show(fig)
    else:
        fig.clear()
        plt.close(fig)


def plot_time_between_files(
    out_directory: str,
    camera_files: Dict[str, List[str]],
    bmjds: Dict[str, float],
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the times between files. Useful for identifying gaps.
    
    Parameters
    ----------
    out_directory : str
        The directory path to which the resulting plot will be saved.
    camera_files : Dict[str, List[str]]
        The file paths separated by camera {filter: file paths}.
    bmjds : Dict[str, float]
        The file time stamps {file path: time stamp}.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    fig, axes = plt.subplots(nrows=3, ncols=len(camera_files), tight_layout=True,
                                figsize=((2 * len(camera_files) / 3) * 6.4, 2 * 4.8), sharey='row')
    
    for fltr in list(camera_files.keys()):
        times = np.array([bmjds[file] for file in camera_files[fltr]])
        times -= times.min()
        times *= 86400  # convert to seconds from first observation
        dt = np.diff(times)  # get time between files
        file_numbers = np.arange(2, len(times) + 1, 1)  # start from 2 because we are plotting the time between files
        
        bin_edges = np.arange(int(dt.min()), np.ceil(dt.max() + .2), .1)  # define bins with width 0.1 s
        
        if len(camera_files) == 1:
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
            axes[0, list(camera_files.keys()).index(fltr)].set_title(fltr)
            
            # cumulative plot of time between files
            axes[0, list(camera_files.keys()).index(fltr)].plot(file_numbers, np.cumsum(dt), "k-", lw=1)
            
            # time between each file
            axes[1, list(camera_files.keys()).index(fltr)].plot(file_numbers, dt, "k-", lw=1)
            
            # histogram of time between files
            axes[2, list(camera_files.keys()).index(fltr)].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
            axes[2, list(camera_files.keys()).index(fltr)].set_yscale("log")
            
            axes[0, 0].set_ylabel("Cumulative time between files [s]")
            axes[1, 0].set_ylabel("Time between files [s]")
            
            for col in range(len(camera_files)):
                axes[0, col].set_xlabel("File number")
                axes[1, col].set_xlabel("File number")
                axes[2, col].set_xlabel("Time between files [s]")
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    
    if save:
        fig.savefig(os.path.join(out_directory, "diag/header_times.png"))
    
    if show:
        plt.show(fig)
    else:
        plt.close(fig)


def plot_backgrounds(
    out_directory: str,
    camera_files: Dict[str, List[str]],
    background_median: Dict[str, Dict[str, NDArray]],
    background_rms: Dict[str, Dict[str, NDArray]],
    bmjds: Dict[str, float],
    t_ref: float,
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the time-varying background for each camera.
    
    Parameters
    ----------
    camera_files : Dict[str, str]
        The files for each camera {fltr: file}.
    background_median : Dict[str, List]
        The median background for each camera.
    background_rms : Dict[str, List]
        The background RMS for each camera.
    bmjds : Dict[str, float]
        The Barycentric MJD dates for each image {file: BMJD}.
    t_ref : float
        The reference BMJD.
    out_directory : str
        The directory to which the resulting files will be saved.
    show: bool
        Whether to display the plot.
    save : bool
        Whether to save the plot.
    """
    
    fig, axs = plt.subplots(
        nrows=2,
        ncols=len(camera_files),
        tight_layout=True,
        figsize=((2 * len(camera_files) / 3) * 6.4, 2 * 4.8),
        sharex='col',
        )
    
    # for each camera
    for fltr in list(camera_files.keys()):
        
        files = camera_files[fltr]  # get files for camera
        
        # skip cameras with no images
        if len(files) == 0:
            continue
        
        # get values from background_median and background_rms dicts
        backgrounds = list(background_median[fltr].values())
        rmss = list(background_rms[fltr].values())
        
        # match times to background_median and background_rms keys
        t = np.array([bmjds[file] for file in files if file in background_median[fltr]])
        plot_times = (t - t_ref) * 86400  # convert time to seconds from first observation
        
        if len(camera_files) == 1:
            axs[0].set_title(fltr)
            axs[0].plot(plot_times, backgrounds, "k.", ms=2)
            axs[1].plot(plot_times, rmss, "k.", ms=2)
            
            axs[1].set_xlabel(f"Time from BMJD {t_ref:.4f} [s]")
            axs[0].set_ylabel("Median background RMS")
            axs[1].set_ylabel("Median background")
        else:
            # plot background
            axs[0, list(camera_files.keys()).index(fltr)].set_title(fltr)
            axs[0, list(camera_files.keys()).index(fltr)].plot(plot_times, backgrounds, "k.", ms=2)
            axs[1, list(camera_files.keys()).index(fltr)].plot(plot_times, rmss, "k.", ms=2)
            
            for col in range(len(camera_files)):
                axs[1, col].set_xlabel(f"Time from BMJD {t_ref:.4f} [s]")
            
            axs[0, 0].set_ylabel("Median background")
            axs[1, 0].set_ylabel("Median background RMS")
        
        # write background to file
        DataFrame({
            'BMJD': t,
            'RMS': backgrounds,
            'median': rmss
        }).to_csv(os.path.join(out_directory, f'diag/{fltr}_background.csv'), index=False)
    
    for ax in axs.flatten():
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    
    if save:
        fig.savefig(os.path.join(out_directory, "diag/background.png"))
    
    if show:
        plt.show()
    else:
        fig.clear()
        plt.close(fig)


def plot_background_meshes(
    out_directory: str,
    filters: List[str],
    stacked_images: Dict[str, NDArray],
    background: BaseBackground,
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the background meshes on top of the catalog images.
    
    Parameters
    ----------
    stacked_images : Dict[str, NDArray]
        The stacked images for each camera.
    show : bool
        Whether to display the plot.
    """
    
    ncols = len(filters)
    fig, axes = plt.subplots(ncols=ncols, tight_layout=True, figsize=(ncols * 5, 5))
    
    if ncols == 1:
        # convert axes to list
        axes = [axes]
    
    for i, fltr in enumerate(filters):
        
        plot_image = np.clip(stacked_images[fltr], 0, None)
        bkg = background(stacked_images[fltr])
        
        # plot background mesh
        axes[i].imshow(plot_image, origin="lower", cmap="Greys_r", interpolation="nearest",
                        norm=simple_norm(plot_image, stretch="log"))
        bkg.plot_meshes(ax=axes[i], outlines=True, marker='.', color='cyan', alpha=0.3)
        
        #label plot
        axes[i].set_title(fltr)
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
    
    if save:
        fig.savefig(os.path.join(out_directory, "diag/background_meshes.png"))
    
    if show:
        plt.show(fig)
    else:
        fig.clear()
        plt.close(fig)