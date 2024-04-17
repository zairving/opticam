from astropy.io import fits
import numpy as np
from numpy.typing import ArrayLike
from photutils.background import Background2D
from typing import Literal, Union, Tuple
from astropy.time import Time
from astropy.stats import SigmaClip
import os


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






