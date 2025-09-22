import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Tuple
import re
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from astropy.visualization import simple_norm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from astropy.table import QTable
import matplotlib.colors as mcolors
import os



def camel_to_snake(
    string: str,
    ) -> str:
    """
    Convert a camelCase string to snake_case.
    
    Parameters
    ----------
    string : str
        The camelCase string to convert.
    
    Returns
    -------
    str
        The converted snake_case string.
    """
    
    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


def plot_catalog(
    filters: List[str],
    stacked_images: Dict[str, NDArray],
    catalogs: Dict[str, QTable],
    ) -> Tuple[Figure, List[Axes]]:
    
    colours = list(mcolors.TABLEAU_COLORS.keys())
    colours.pop(colours.index("tab:brown"))
    colours.pop(colours.index("tab:gray"))
    colours.pop(colours.index("tab:purple"))
    colours.pop(colours.index("tab:blue"))
    
    fig, axes = plt.subplots(
        ncols=len(filters),
        tight_layout=True,
        figsize=(
            len(stacked_images) * 5,
            5,
            ),
        )
    
    if len(filters) == 1:
        axes = [axes]
    
    for i, fltr in enumerate(filters):
        
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
                    edgecolor=colours[j % len(colours)],
                    facecolor="none",
                    lw=1,
                    ),
                )
            axes[i].text(
                catalogs[fltr]["xcentroid"][j] + 1.05 * radius,
                catalogs[fltr]["ycentroid"][j] + 1.05 * radius,
                j + 1,  # source number
                color=colours[j % len(colours)],
                )
            
            # label plot
            axes[i].set_title(fltr)
            axes[i].set_xlabel("X")
            
            if i > 0:
                axes[i].set_ylabel("Y")
    
    return fig, axes



def sort_filters(
    d: Dict[str, Any],
    ) -> Dict[str, Any]:
    """
    Sort a dictionary whose keys are filter names in the order of the camera filters (e.g., u/g, r, i/z).
    
    Parameters
    ----------
    d : Dict[str, Any]
        A dictionary with filter names as keys.
    
    Returns
    -------
    Dict[str, Any]
        The sorted dictionary.
    """
    
    key_order = {
        'u-band': 0,
        "u'-band": 0,
        'g-band': 0,
        "g'-band": 0,
        "r-band": 1,
        "r'-band": 1,
        'i-band': 2,
        "i'-band": 2,
        'z-band': 2,
        "z'-band": 2,
        }
    
    return dict(sorted(d.items(), key=lambda x: key_order[x[0]]))


def create_file_paths(
    data_directory: None | str = None,
    c1_directory: None | str = None,
    c2_directory: None | str = None,
    c3_directory: None | str = None,
    ) -> List[str]:
    """
    Given some directories, get the paths to all available FITS files.
    
    Parameters
    ----------
    data_directory : None | str, optional
        The directory containing the FITS files of all three cameras, by default None.
    c1_directory : None | str, optional
        The directory containing the FITS files of Camera 1, by default None.
    c2_directory : None | str, optional
        The directory containing the FITS files of Camera 2, by default None.
    c3_directory : None | str, optional
        The directory containing the FITS files of Camera 3, by default None.
    
    Returns
    -------
    List[str]
        The file paths.
    """
    
    file_paths = []
    
    for directory in [data_directory, c1_directory, c2_directory, c3_directory]:
        if directory is not None:
            file_names = os.listdir(directory)
            for file_name in file_names:
                if '.fit' in file_name:
                    file_paths.append(os.path.join(directory, file_name))
    
    return file_paths











