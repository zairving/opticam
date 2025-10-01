from logging import Logger
from typing import Dict, Tuple

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from ccdproc import cosmicray_lacosmic  # TODO: replace with astroscrappy to reduce dependencies?
import numpy as np
from numpy.typing import ArrayLike, NDArray
import os.path

from opticam.correctors.flat_field_corrector import FlatFieldCorrector
from opticam.utils.time_helpers import apply_barycentric_correction
from opticam.utils.image_helpers import rebin_image


def get_header_info(
    file: str,
    logger: Logger | None,
    ) -> Tuple[ArrayLike | None, str | None, str | None, float | None]:
    """
    Get the BMJD, filter, binning, and gain from a file header.
    
    Parameters
    ----------
    file : str
        The file path.
    
    Returns
    -------
    Tuple[float, str, str, float]
        The BMJD, filter, binning, and gain dictionaries.
    """
    
    try:
        with fits.open(file) as hdul:
            header = hdul[0].header
        
        binning = header["BINNING"]
        gain = header["GAIN"]
        
        try:
            ra = header["RA"]
            dec = header["DEC"]
        except:
            if logger:
                logger.info(f"[OPTICAM] Could not find RA and DEC keys in {file} header.")
            pass
        
        mjd = get_time(header, file)
        fltr = header["FILTER"]
        
        try:
            # try to compute barycentric dynamical time
            coords = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            bmjd = apply_barycentric_correction(mjd, coords)
        except Exception as e:
            if logger:
                logger.info(f"[OPTICAM] Could not compute BMJD for {file}: {e}. Skipping.")
            return None, None, None, None
    except Exception as e:
        if logger:
            logger.info(f'[OPTICAM] Could not read {file}: {e}. Skipping.')
        return None, None, None, None
    
    return bmjd, fltr, binning, gain


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


def get_data(
    file: str,
    gain: float,
    flat_corrector: FlatFieldCorrector | None,
    rebin_factor: int,
    return_error: bool,
    remove_cosmic_rays: bool,
    ) -> NDArray | Tuple[NDArray, NDArray]:
    """
    Get the image data from a FITS file.
    
    Parameters
    ----------
    file : str
        The file.
    gain : float
        The file gain.
    flat_corrector : FlatFieldCorrector | None
        The `FlatFieldCorrector` instance (if specified).
    rebin_factor : int
        The rebin factor.
    return_error : bool
        Whether to compute and return the error image.
    remove_cosmic_rays : bool
        Whether to remove cosmic rays from the image.
    
    Returns
    -------
    NDArray | Tuple[NDArray, NDArray]
        _description_
    
    Raises
    ------
    ValueError
        If `file` could not be opened.
    """
    
    try:
        with fits.open(file) as hdul:
            data = np.array(hdul[0].data, dtype=np.float64)
            fltr = hdul[0].header["FILTER"] + '-band'
    except:
        raise ValueError(f"[OPTICAM] Could not open file {file}.")
    
    if return_error:
        error = np.sqrt(data * gain)
    
    if flat_corrector:
        data = flat_corrector.correct(data, fltr)
        
        if return_error:
            error = flat_corrector.correct(error, fltr)
    
    # remove cosmic rays if required
    if remove_cosmic_rays:
        data = np.asarray(cosmicray_lacosmic(data, gain_apply=False)[0])
    
    if rebin_factor > 1:
        data = rebin_image(data, rebin_factor)
        
        if return_error:
            error = rebin_image(error, rebin_factor)
    
    if return_error:
        return data, error
    
    return data


def save_stacked_images(
    stacked_images: Dict[str, NDArray],
    out_directory: str,
    overwrite: bool,
    ) -> None:
    """
    Save the stacked images to a compressed FITS file.
    
    Parameters
    ----------
    stacked_images : Dict[str, NDArray]
        The stacked images (filter: stacked image).
    """
    
    hdr = fits.Header()
    hdr['COMMENT'] = 'This FITS file contains the stacked images for each filter.'
    empty_primary = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([empty_primary])
    
    for fltr, img in stacked_images.items():
        hdr = fits.Header()
        hdr['FILTER'] = fltr
        hdu = fits.ImageHDU(img, hdr)
        hdul.append(hdu)
    
    file_path = os.path.join(out_directory, f'cat/stacked_images.fits.gz')
    
    if not os.path.isfile(file_path) or overwrite:
        hdul.writeto(file_path, overwrite=overwrite)


def get_stacked_images(
    out_directory: str,
    ) -> Dict[str, NDArray]:
    """
    Unpacked the stacked catalog images from out_directory/cat.
    
    Parameters
    ----------
    out_directory : str
        The directory path to the reduction output.
    
    Returns
    -------
    Dict[str, NDArray]
        The stacked images {filter: image}.
    """
    
    stacked_images = {}
    with fits.open(os.path.join(out_directory, 'cat/stacked_images.fits.gz')) as hdul:
        for hdu in hdul:
            if 'FILTER' not in hdu.header:
                continue
            fltr = hdu.header['FILTER']
            stacked_images[fltr] = np.asarray(hdu.data)
    
    return stacked_images