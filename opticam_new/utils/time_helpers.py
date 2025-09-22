from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
from astropy import units as u
import numpy as np
from numpy.typing import NDArray


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