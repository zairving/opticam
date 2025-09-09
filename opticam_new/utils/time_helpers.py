from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u
import numpy as np
from numpy.typing import NDArray
from typing import Dict


def apply_barycentric_correction(
    original_times: float | NDArray,
    coords: SkyCoord,
    ) -> float | NDArray:
    """
    Apply barycentric corrections to a time array.
    
    Parameters
    ----------
    times : float | NDArray
        The time(s) to correct.
    coords : SkyCoord
        The coordinates of the source.
    
    Returns
    -------
    float | NDArray
        The corrected time(s).
    """
    
    # OPTICAM location
    observer_coords = EarthLocation.from_geodetic(lon=-115.463611*u.deg, lat=31.044167*u.deg, height=2790*u.m)
    
    # format the times
    times = Time(original_times, format='mjd', scale='utc', location=observer_coords)
    
    # compute light travel time to barycentre
    ltt_bary = times.light_travel_time(coords)
    
    return (times.tdb + ltt_bary).value


def get_time(
    header: Dict,
    file: str,
    ) -> float:
    """
    Parse the time from the header of a FITS file.
    
    Parameters
    ----------
    header
        The FITS file header.
    file : str
        The path to the file.
    
    Returns
    -------
    float
        The time of the observation in MJD.
    
    Raises
    ------
    ValueError
        If the time cannot be parsed from the header.
    KeyError
        If neither 'GPSTIME' nor 'UT' keys are found in the header.
    """
    
    if "GPSTIME" in header.keys():
        gpstime = header["GPSTIME"]
        split_gpstime = gpstime.split(" ")
        date = split_gpstime[0]  # get date
        time = split_gpstime[1].split(".")[0]  # get time (ignoring decimal seconds)
        mjd = Time(date + "T" + time, format="fits").mjd
    elif "UT" in header.keys():
        try:
            mjd = Time(header["UT"].replace(" ", "T"), format="fits").mjd
        except:
            try:
                date = header['DATE-OBS']
                time = header['UT'].split('.')[0]
                mjd = Time(date + 'T' + time, format='fits').mjd
            except:
                raise ValueError('Could not parse time from ' + file + ' header.')
    else:
        raise KeyError(f"[OPTICAM] Could not find GPSTIME or UT key in {file} header.")
    
    return mjd


def infer_gtis(time: NDArray, threshold: float = 1.5) -> NDArray:
    """
    Infer GTIs from a light curve.
    
    Parameters
    ----------
    time : ArrayLike
        The time array.
    threshold : float, optional
        The threshold for detecting gaps in units of the median time resolution, by default 1.5.
    
    Returns
    -------
    List[Tuple[float, float]]
        The inferred GTIs.
    """
    
    time = np.asarray(time)
    
    # compute the gap threshold
    gap_threshold = threshold * np.median(np.diff(time))
    
    # define GTI starts and stops
    gti_starts = [time[0]]
    gti_stops = []
    
    # compute GTIs
    for i in range(1, time.size):
        if time[i] - time[i - 1] > gap_threshold:
            gti_stops.append(time[i - 1])
            gti_starts.append(time[i])
    gti_stops.append(time[-1])
    
    # define GTIs in stingray format
    return np.array(list(zip(gti_starts, gti_stops)))