from astropy.io import fits
import numpy as np
from numpy.typing import ArrayLike
from typing import Any, List, Tuple
from astropy.io import fits
import json
import re
from types import FunctionType


# custom tqdm progress bar format
bar_format= '{l_bar}{bar}|[{elapsed}<{remaining}]'

# camera pixel scales
pixel_scales = {
    'u-band': 0.1397, 'g-band': 0.1397,
    'r-band': 0.1406,
    'i-band': 0.1661, 'z-band': 0.1661,
    }

# stdev -> FWHM scale factor
fwhm_scale = 2 * np.sqrt(2 * np.log(2))


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


def euclidean_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    ) -> float:
    """
    Compute the Euclidean distance between two points.
    
    Parameters
    ----------
    p1 : Tuple[float, float]
        The x and y coordinates of the first point.
    p2 : Tuple[float, float]
        The x and y coordinates of the second point.
    
    Returns
    -------
    float
        The distance.
    """
    
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def find_closest_pair(
    point: Tuple[float, float],
    points: List[Tuple[float, float]],
    threshold: int,
    ) -> ArrayLike | None:
    
    distances = [(euclidean_distance(point, point2), point2) for point2 in points]  # compute distances
    
    distances.sort(key=lambda x: x[0])  # sort by distance
    
    if distances[0][0] > threshold:
        return None
    
    # return the closest pair
    return distances[0][1]


def log_binnings(
    file_paths: List[str],
    out_directory: str,
    ) -> None:
    """
    Log the binning of each file to out_directory/diag/binnings.json.
    
    Parameters
    ----------
    file_paths : List[str]
        The paths to the files.
    out_directory : str
        The directory to save the log.
    """
    
    file_binnings = {}
    
    for file in file_paths:
        with fits.open(file) as hdul:
            binning = hdul[0].header["BINNING"]
            if binning in file_binnings:
                file_binnings[binning].append(file)
            else:
                file_binnings[binning] = [file]
    
    with open(out_directory + "diag/binnings.json", "w") as f:
        json.dump(file_binnings, f, indent=4)


def log_filters(
    file_paths: List[str],
    out_directory: str,
    ) -> None:
    """
    Logs the filters used in each file to out_directory/diag/filters.json.
    
    Parameters
    ----------
    file_paths : List[str]
        The paths to the files.
    out_directory : str
        The directory to save the log.
    """
    
    file_filters = {}
    
    for file in file_paths:
        with fits.open(file) as hdul:
            fltr = hdul[0].header["FILTER"]
            if fltr in file_filters:
                file_filters[fltr].append(file)
            else:
                file_filters[fltr] = [file]
    
    with open(out_directory + "diag/filters.json", "w") as f:
        json.dump(file_filters, f, indent=4)


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