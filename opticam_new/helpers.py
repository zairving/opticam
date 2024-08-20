from astropy.io import fits
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, Tuple
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
import os
from astropy.io import fits
import json
import itertools
from astropy.table import QTable




def get_data(file: str) -> ArrayLike:
    """
    Read 

    Parameters
    ----------
    file : str
        Directory path to file.

    Returns
    -------
    ArrayLike
        Image data as an np.ndarray.
    """
    
    try:
        with fits.open(file) as hdul:
            data = np.array(hdul[0].data, dtype=np.float64)
    except:
        raise ValueError(f"[OPTICAM] Could not open file {file}.")
    
    return data


def get_time(file: str, date_key: Literal["UT", "GPSTIME"]) -> float:
    """
    Parse the time from the header of a FITS file.

    Parameters
    ----------
    file : str
        Directory path to file.
    date_key : Literal[&quot;UT&quot;, &quot;GPSTIME&quot;]
        Header key that gives the observation date and time.

    Returns
    -------
    float
        The time of the observation in MJD.

    Raises
    ------
    KeyError
        _description_
    KeyError
        _description_
    ValueError
        _description_
    """
    
    with fits.open(file) as hdul:
        if date_key == "UT":
            try:
                time = Time(hdul[0].header["UT"].replace(" ", "T"), format="fits").mjd
            except:
                raise KeyError(f"[OPTICAM] Could not find {date_key} key in header.")
        elif date_key == "GPSTIME":
            try:
                temp = hdul[0].header["GPSTIME"]
            except:
                raise KeyError(f"[OPTICAM] Could not find {date_key} key in header.")
            try:
                split_temp = temp.split(" ")
                date = split_temp[0]
                gps_time = split_temp[1].split(".")[0]  # remove decimal seconds
                time = Time(date + "T" + gps_time, format="fits").mjd
            except:
                raise ValueError(f"[OPTICAM] Could not parse {date_key} key in header.")

    return time


def log_binnings(data_directory: str, out_directory: str):
    
    file_binnings = {}
    
    for file in sorted(os.listdir(data_directory)):
        with fits.open(data_directory + file) as hdul:
            binning = hdul[0].header["BINNING"]
            if binning in file_binnings:
                file_binnings[binning].append(file)
            else:
                file_binnings[binning] = [file]
    
    with open(out_directory + "diag/binnings.json", "w") as f:
        json.dump(file_binnings, f, indent=4)


def log_filters(data_directory: str, out_directory: str):
    
    file_filters = {}
    
    for file in sorted(os.listdir(data_directory)):
        with fits.open(data_directory + file) as hdul:
            fltr = hdul[0].header["FILTER"]
            if fltr in file_filters:
                file_filters[fltr].append(file)
            else:
                file_filters[fltr] = [file]
    
    with open(out_directory + "diag/filters.json", "w") as f:
        json.dump(file_filters, f, indent=4)


def default_aperture_selector(aperture_sizes: ArrayLike) -> float:
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


def apply_barycentric_correction(original_times: ArrayLike, coords: SkyCoord) -> ArrayLike:
    """
    Apply barycentric corrections to a time vector, using an optional reference time.
    
    Parameters
    ----------
    times : ArrayLike
        The times to correct.
    coords : SkyCoord
        The coordinates of the source.
    
    Returns
    -------
    ArrayLike
        The corrected times.
    """
    
    # OPTICam location
    observer_coords = EarthLocation.from_geodetic(lon=-115.463611*u.deg, lat=31.044167*u.deg, height=2790*u.m)
    
    
    # format the times
    times = Time(original_times, format='mjd', scale='utc', location=observer_coords)
    
    # compute light travel time to barycentre
    ltt_bary = times.light_travel_time(coords)
    
    # return the corrected times using TDB timescale
    return times.tdb + ltt_bary


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
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


def find_closest_pair(point: ArrayLike, points: ArrayLike, threshold: int) -> ArrayLike:
    
    distances = [(euclidean_distance(point, point2), point2) for point2 in points]  # compute distances
    
    distances.sort(key=lambda x: x[0])  # sort by distance
    
    if distances[0][0] > threshold:
        return None
    
    # return the closest pair
    return distances[0][1]


def clip_extended_sources(table: QTable):
    
    # sigma clip sources
    for i in range(10):
        median_radius = np.median(table["semimajor_sigma"].value)
        radius_std = np.std(table["semimajor_sigma"].value)
        table = table[table['semimajor_sigma'].value <= median_radius + 3*radius_std]  # remove sources with radius > 3*std from median
    
    median_radius = np.median(table["semimajor_sigma"].value)  # get median source radius
    radius_std = np.std(table["semimajor_sigma"].value)  # get standard deviation of source radius
    
    table = table[table['semimajor_sigma'].value <= median_radius + 3*radius_std]  # remove sources with radius > 3*std from median
    
    table['label'] = np.arange(1, len(table) + 1)  # renumber labels
    
    return table


def rebin_image(image: NDArray, factor: int) -> NDArray:
    """
    Rebin an image by a given factor.
    
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
        raise ValueError("[OPTICAM] The dimensions of the input data must be divisible by the rebinning factor.")
    
    # reshape the array to make it ready for block summation
    shape = (image.shape[0] // factor, factor, image.shape[1] // factor, factor)
    reshaped_data = image.reshape(shape)
    
    # return rebinned image
    return reshaped_data.sum(axis=(1, 3))