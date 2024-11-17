from astropy.io import fits
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Literal, List, Tuple
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
import os
from astropy.io import fits
import json
from astropy.table import QTable
from tqdm import tqdm


def get_time(hdul, file: str) -> float:
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
        _description_
    KeyError
        _description_
    ValueError
        _description_
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


def log_binnings(file_paths: List[str], out_directory: str) -> None:
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


def log_filters(file_paths: List[str], out_directory: str) -> None:
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


def identify_gaps(files: List[str], log_dir: str):
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
            file_times[file] = get_time(hdul, file)
    
    sorted_files = dict(sorted(file_times.items(), key=lambda x: x[1]))
    times = np.array(sorted_files.values()).flatten()
    diffs = np.diff(times)
    median_exposure_time = np.median(diffs)
    
    gaps = np.where(diffs > 2*median_exposure_time)[0]
    
    if len(gaps) > 0:
        print(f"[OPTICAM] Found {len(gaps)} gaps in the observation sequence.")
        with open(log_dir + "diag/gaps.txt", "w") as file:
            file.write(f"Median exposure time: {median_exposure_time} d\n")
            for gap in gaps:
                file.write(f"Gap between {list(sorted_files.keys())[gap]} and {list(sorted_files.keys())[gap + 1]}: {diffs[gap]} d\n")
    else:
        print("[OPTICAM] No gaps found in the observation sequence.")