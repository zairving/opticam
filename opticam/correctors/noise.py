from copy import copy, deepcopy
from typing import Callable, Dict
import warnings

from astropy.units import Quantity
from astropy.table import QTable
from astropy import units as u
import numpy as np
from numpy.typing import NDArray

from opticam.background.global_background import BaseBackground
from opticam.photometers import AperturePhotometer
from opticam.utils.fits_handlers import get_image_noise_info


def get_source_photons(
    N_source: Quantity,
    gain: Quantity,
    ) -> float:
    """
    Get the number of source photons.
    
    Parameters
    ----------
    N_source : Quantity
        The number of source counts [ADU].
    gain : Quantity
        The gain [photons/ADU].
    
    Returns
    -------
    float
        The number of source photons.
    """
    
    return (N_source * gain).to_value(u.ph)

def get_sky_photons_per_pixel(
    n_sky: Quantity,
    gain: Quantity,
    ) -> float:
    """
    Get the number of sky photons per pixel.
    
    Parameters
    ----------
    n_sky : Quantity
        The sky counts per unit pixel [ADU/pix].
    gain : Quantity
        The gain [photons/ADU].
    
    Returns
    -------
    float
        The number of sky photons per pixel.
    """
    
    return (n_sky * gain).to_value(u.ph / u.pix)

def get_dark_counts_per_pixel(
    dark_curr: Quantity,
    t_exp: Quantity,
    ) -> float:
    """
    Get the number of dark counts per pixel.
    
    Parameters
    ----------
    dark_curr : Quantity
        The dark current [ADU/pixel/second]
    t_exp : Quantity
        The exposure time [seconds].
    
    Returns
    -------
    float
        The number of dark counts per pixel.
    """
    
    return (dark_curr * t_exp).to_value(u.adu / u.pix)

def get_read_counts_per_pixel(
    n_read: Quantity,
    ) -> float:
    """
    Get the read noise per pixel.
    
    Parameters
    ----------
    n_read : Quantity
        The number of counts due to read noise per pixel [ADU/pixel].
    
    Returns
    -------
    float
        The read noise per pixel.
    """
    
    return n_read.to_value(u.adu / u.pix)


def get_sky_stderr(
    N_source: Quantity,
    N_pix: int,
    n_sky: Quantity,
    gain: Quantity,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the sky noise.
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    n_sky : Quantity
        The number of sky counts **per pixel**.
    gain: Quantity
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the sky noise.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    sky_photons_per_pix = get_sky_photons_per_pixel(n_sky=n_sky, gain=gain)
    
    p_sky = N_pix.value * sky_photons_per_pix
    
    return 1.0857 * np.sqrt(p_sky) / source_photons

def get_shot_stderr(
    N_source: Quantity,
    gain: Quantity,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the shot noise.
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    gain: Quantity
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the shot noise.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    
    return 1.0857 * np.sqrt(source_photons) / source_photons

def get_dark_stderr(
    N_source: Quantity,
    N_pix: int,
    n_dark: Quantity,
    t_exp: Quantity,
    gain: Quantity,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the dark current noise.
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    dark_curr : Quantity
        The number of dark current electrons **per pixel per unit time**. 
    t_exp: Quantity
        The exposure time.
    gain: Quantity
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the dark current noise.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    dark_counts_per_pixel = get_dark_counts_per_pixel(dark_curr=n_dark, t_exp=t_exp)
    
    p_dark = N_pix.value * dark_counts_per_pixel
    
    return 1.0857 * np.sqrt(p_dark) / source_photons

def get_read_stderr(
    N_source: Quantity,
    N_pix: int,
    n_read: Quantity,
    gain: Quantity,
    ) -> float:
    """
    Get the standard error (in magnitudes) of the readout noise.
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    n_read : Quantity
        The number of electrons **per pixel** due to read noise.
    gain: Quantity
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) of the readout noise.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    read_counts_per_pixel = get_read_counts_per_pixel(n_read=n_read)
    
    p_read = N_pix.value * read_counts_per_pixel**2
    
    return 1.0857 * np.sqrt(p_read) / source_photons


def snr(
    N_source: Quantity,
    N_pix: int,
    n_sky: Quantity,
    dark_curr: Quantity,
    n_read: Quantity,
    t_exp: Quantity,
    gain: Quantity,
    ) -> float:
    """
    The simplified S/N ratio equation or CCD Equation (see Chapter 4.4 of Handbook of CCD Astronomy by Howell, 2006).
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    n_sky : Quantity
        The number of sky counts **per pixel**.
    dark_curr : Quantity
        The number of dark current electrons **per pixel per unit time**. 
    n_read : Quantity
        The number of electrons **per pixel** due to read noise.
    t_exp: Quantity
        The exposure time.
    gain: Quantity
        The detector gain.
    
    Returns
    -------
    float
        The S/N ratio.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    sky_photons_per_pix = get_sky_photons_per_pixel(n_sky=n_sky, gain=gain)
    dark_counts_per_pixel = get_dark_counts_per_pixel(dark_curr=dark_curr, t_exp=t_exp)
    read_counts_per_pixel = get_read_counts_per_pixel(n_read=n_read)
    
    return source_photons / np.sqrt(source_photons + N_pix.value * (sky_photons_per_pix + dark_counts_per_pixel + read_counts_per_pixel**2))

def snr_stderr(
    N_source: Quantity,
    N_pix: int,
    n_sky: Quantity,
    dark_curr: Quantity,
    n_read: Quantity,
    t_exp: Quantity,
    gain: Quantity,
    ) -> float:
    """
    The standard error (in magnitudes) on the CCD Equation (see Chapter 4.4 of Handbook of CCD Astronomy by Howell, 
    2006).
    
    Parameters
    ----------
    N_source : Quantity
        The total number of source counts.
    N_pix : int
        The number of aperture pixels.
    n_sky : Quantity
        The number of sky counts **per pixel**.
    dark_curr : Quantity
        The number of dark current electrons **per pixel per unit time**. 
    n_read : Quantity
        The number of electrons **per pixel** due to read noise.
    t_exp: Quantity
        The exposure time.
    gain: Quantity
        The detector gain.
    
    Returns
    -------
    float
        The standard error (in magnitudes) on the S/N ratio.
    """
    
    source_photons = get_source_photons(N_source=N_source, gain=gain)
    sky_photons_per_pix = get_sky_photons_per_pixel(n_sky=n_sky, gain=gain)
    dark_counts_per_pixel = get_dark_counts_per_pixel(dark_curr=dark_curr, t_exp=t_exp)
    read_counts_per_pixel = get_read_counts_per_pixel(n_read=n_read)
    
    p = N_pix.value * (sky_photons_per_pix + dark_counts_per_pixel + read_counts_per_pixel**2)
    
    return 1.0857 * np.sqrt(source_photons + p) / source_photons


def characterise_noise(
    file: str,
    background: BaseBackground | Callable,
    catalog: QTable,
    psf_params: Dict[str, float],
    ) -> Dict[str, NDArray]:
    
    n_read = 1.1 * u.adu / u.pix  # Andor Zyla 4.2 PLUS @ 216 MHz
    
    coords = np.asarray([catalog['xcentroid'], catalog['ycentroid']]).T
    
    img, t_exp, dark_curr, gain = get_image_noise_info(file)
    
    # global background
    bkg = background(img)
    n_sky = bkg.background_median * u.adu / u.pix
    
    img_clean = img - bkg.background
    img_clean_err = np.sqrt(img + bkg.background_rms**2)
    
    phot = AperturePhotometer()
    phot_results = phot.compute(
        img_clean,
        img_clean_err,
        coords,
        coords,
        psf_params,
        )
    
    # get the effective number of pixels in the aperture
    N_pix = phot.get_aperture_area(psf_params=psf_params) * u.pix
    
    fluxes = np.asarray([flux for flux in phot_results['flux']])
    flux_errs = np.asarray([flux for flux in phot_results['flux_err']])
    
    N_source = np.logspace(
        np.log10(np.min(fluxes) / 1.5),
        np.log10(np.max(fluxes) * 1.5),
        100,
        ) * u.adu
    
    results = {}
    
    results['model_mags'] = -2.5 * np.log10(N_source.value)
    results['effective_noise'] = snr_stderr(N_source, N_pix, n_sky, dark_curr, n_read, t_exp, gain)
    results['sky_noise'] = get_sky_stderr(N_source, N_pix, n_sky, gain)
    results['shot_noise'] = get_shot_stderr(N_source, gain)
    results['dark_noise'] = get_dark_stderr(N_source, N_pix, dark_curr, t_exp, gain)
    results['read_noise'] = get_read_stderr(N_source, N_pix, n_read, gain)
    
    results['measured_mags'] = -2.5 * np.log10(fluxes)
    results['measured_noise'] = 1.0857 * flux_errs / fluxes,
    results['expected_measured_noise'] = snr_stderr(fluxes * u.adu, N_pix, n_sky, dark_curr, n_read, t_exp, gain)
    
    return results