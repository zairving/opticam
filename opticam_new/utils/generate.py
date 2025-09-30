from astropy.io import fits
import os
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from typing import List

from opticam_new.utils.constants import bar_format


def _add_two_dimensional_gaussian_to_image(
    image: NDArray,
    x_centroid: float,
    y_centroid: float,
    peak_flux: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    ) -> NDArray:
    """
    Add a source to an image.
    
    Parameters
    ----------
    image : NDArray
        The image.
    x_centroid : float
        The x-coordinate of the source.
    y_centroid : float
        The y-coordinate of the source.
    peak_flux : float
        The peak flux of the source.
    sigma_x : float
        The standard deviation of the source in the x-direction.
    sigma_y : float
        The standard deviation of the source in the y-direction.
    theta : float
        The rotation angle of the source.
    
    Returns
    -------
    NDArray
        The image with the source added.
    """
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    
    gaussian = peak_flux*np.exp(-(a*(x - x_centroid)**2 + 2*b*(x - x_centroid)*(y - y_centroid) + c*(y - y_centroid)**2))
    
    return image + gaussian

def _variable_function(
    i: float,
    fltr: str,
    ) -> float:
    """
    Variable flux to be added to a source.
    
    Parameters
    ----------
    i : float
        The image index (equivalent to time).
    fltr : str
        The filter of the image, used to introduce a lag between filters.
    
    Returns
    -------
    float
        The flux.
    """
    
    if fltr == 'g':
        phase = 0
    elif fltr == 'r':
        phase = - np.pi / 2
    elif fltr == 'i':
        phase = - np.pi
    
    return 20 * np.sin(2 * np.pi * i * 0.135 + phase)

def _create_image(
    binning_scale: int,
    ) -> NDArray:
    """
    Create a base image with Poisson noise.
    
    Parameters
    ----------
    binning_scale : int
        The binning scale of the image.
    
    Returns
    -------
    NDArray
        The noisy image.
    """
    
    return np.zeros((int(2048 / binning_scale), int(2048 / binning_scale))) + 100

def _add_noise(
    image: NDArray,
    i: int,
    ) -> NDArray:
    """
    Add Poisson noise to an image.
    
    Parameters
    ----------
    image : NDArray
        The image to add noise to.
    i : int
        The image index, used to seed the random number generator.
    
    Returns
    -------
    NDArray
        The noisy image.
    """
    
    rng = np.random.default_rng(i)
    
    return rng.normal(image, np.sqrt(image))

def _create_images(
    out_dir: str,
    filters: List[str],
    N_sources: int,
    variable_source: int,
    source_positions: NDArray,
    peak_fluxes: NDArray,
    i: int,
    binning_scale: int,
    circular_aperture: bool,
    overwrite: bool,
    ) -> None:
    """
    Create the ith image for each filter.
    
    Parameters
    ----------
    filters : List[str]
        The filters.
    N_sources : int
        The number of sources.
    variable_source : int
        The index of the variable source.
    source_positions : NDArray
        The positions of the sources.
    peak_fluxes : NDArray
        The peak fluxes of the sources.
    i : int
        The image index (equivalent to time).
    binning_scale : int
        The binning scale of the image.
    circular_aperture : bool
        Whether to apply a circular aperture shadow to the image.
    overwrite : bool
        Whether to overwrite the image if it already exists.
    """
    
    for fltr in filters:
        if os.path.isfile(f"{out_dir}/240101{fltr}{200000000 + i}o.fits.gz") and not overwrite:
            continue
        
        # generate image
        image = _create_image(binning_scale)
        if circular_aperture:
            image = _apply_flat_field(image)  # apply circular aperture shadow
        noisy_image = _add_noise(image, i)  # add Poisson noise
        
        # PSF parameters
        semimajor_sigma = noisy_image.shape[0] // 256
        semiminor_sigma = noisy_image.shape[1] // 256
        orientation = 0
        
        # (x, y) translations
        rng = np.random.default_rng(i)
        dx = rng.normal()
        dy = rng.normal()
        x_positions = source_positions[:, 0] + dx
        y_positions = source_positions[:, 1] + dy
        
        # put sources in the image
        for j in range(N_sources):
            
            if j == variable_source:
                noisy_image = _add_two_dimensional_gaussian_to_image(
                    noisy_image,
                    x_positions[j],
                    y_positions[j],
                    peak_fluxes[j] + _variable_function(i, fltr),  # add variable flux to the source
                    semimajor_sigma,
                    semiminor_sigma,
                    orientation,
                    )
            else:
                noisy_image = _add_two_dimensional_gaussian_to_image(
                    noisy_image,
                    x_positions[j],
                    y_positions[j],
                    peak_fluxes[j],
                    semimajor_sigma,
                    semiminor_sigma,
                    orientation,
                    )
        
        # create fits file
        hdu = fits.PrimaryHDU(noisy_image)
        hdu.header["FILTER"] = fltr
        hdu.header["BINNING"] = f'{binning_scale}x{binning_scale}'
        hdu.header["GAIN"] = 1.
        
        # create observation pointing
        hdu.header['RA'] = 0.
        hdu.header['DEC'] = 0.
        
        # create observation time
        hh = str(i // 3600).zfill(2)
        mm = str(i % 3600 // 60).zfill(2)
        ss = str(i % 60).zfill(2)
        hdu.header["UT"] = f"2024-01-01 {hh}:{mm}:{ss}"
        
        # save fits file
        try:
            hdu.writeto(
                f"{out_dir}/240101{fltr}{200000000 + i}o.fits.gz",
                overwrite=overwrite,
                )
        except:
            pass

def _apply_flat_field(
    image: NDArray,
    ) -> NDArray:
    """
    Apply a circular aperture shadow to an image.
    
    Parameters
    ----------
    image : NDArray
        The image.
    
    Returns
    -------
    NDArray
        The image with a circular aperture shadow.
    """
    
    # define mask to apply circular aperture
    x_mid, y_mid = image.shape[1] // 2, image.shape[0] // 2
    distance_from_centre = np.sqrt((x_mid - np.arange(image.shape[1]))**2 +
                                   (y_mid - np.arange(image.shape[0]))[:, np.newaxis]**2)
    radius = image.shape[0] // 2
    mask = distance_from_centre >= radius
    
    # create circular aperture shadow
    falloff = 1 / (distance_from_centre[mask] / radius)**2
    
    # apply circular aperture shadow
    image[mask] *= falloff
    
    return image

def _create_flats(
    out_dir: str,
    filters: list,
    i: int,
    binning_scale: int,
    overwrite: bool,
    ) -> None:
    """
    Create the ith flat-field image for each filter. 
    
    Parameters
    ----------
    out_dir : str
        The directory to save the flat-field images.
    filters : list
        The filters to create flat-field images for.
    i : int
        The index of the flat-field image (equivalent to time).
    binning_scale : int
        The binning scale of the flat-field image.
    overwrite : bool
        Whether to overwrite the flat-field image if it already exists.
    """
    
    for fltr in filters:
        
        if os.path.isfile(f"{out_dir}/{fltr}-band_image_{i}.fits.gz") and not overwrite:
            continue
        
        # create flat-field image
        image = _create_image(binning_scale)
        image = _apply_flat_field(image)  # apply circular aperture shadow
        noisy_image = _add_noise(image, 123 * (i + 123))  # ensure different noise from observation images
        
        # create fits file
        hdu = fits.PrimaryHDU(noisy_image)
        hdu.header["FILTER"] = fltr
        hdu.header["BINNING"] = f'{binning_scale}x{binning_scale}'
        hdu.header["GAIN"] = 1.
        
        # create observation time
        hh = str(i // 3600).zfill(2)
        mm = str(i % 3600 // 60).zfill(2)
        ss = str(i % 60).zfill(2)
        hdu.header["UT"] = f"2024-01-01 {hh}:{mm}:{ss}"
        
        # save fits file
        try:
            hdu.writeto(f"{out_dir}/{fltr}-band_flat_{i}.fits.gz", overwrite=overwrite)
        except:
            pass

def generate_flats(
    out_dir: str,
    n_flats: int = 5,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic flat-field images.
    
    Parameters
    ----------
    out_dir : str
        The directory to save the data.
    n_flats : int, optional
        The number of flats per camera, by default 5.
    binning_scale : int, optional
        The binning scale of the flat-field images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    # create directory if it does not exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    filters = ["g", "r", "i"]
    
    for i in tqdm(range(n_flats), desc="Generating flats", bar_format=bar_format):
        _create_flats(
            out_dir,
            filters,
            i,
            binning_scale,
            overwrite,
            )

def generate_observations(
    out_dir: str,
    n_images: int = 100,
    circular_aperture: bool = True,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic observation data for testing and following the tutorials.
    
    Parameters
    ----------
    out_dir : str
        The directory to save the data.
    n_images : int, optional
        The number of images to create, by default 100.
    circular_aperture : bool, optional
        Whether to apply a circular aperture shadow to the images, by default True.
    binning_scale : int, optional
        The binning scale of the images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    # create directory if it does not exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    rng = np.random.default_rng(123)
    filters = ["g", "r", "i"]
    
    N_sources = 6
    source_positions = rng.uniform(0 + int(64 / binning_scale), int(2048 / binning_scale - 64 / binning_scale),
                                   (N_sources, 2))  # generate random source positions away from the edges
    peak_fluxes = rng.uniform(100, 1000, N_sources)  # generate random peak fluxes
    variable_source = 1  # index of the variable source
    
    print(f'[OPTICAM] variable source is at ({source_positions[variable_source][0]:.0f}, {source_positions[variable_source][1]:.0f})')
    
    for i in tqdm(range(n_images), desc="Generating observations", bar_format=bar_format):
        _create_images(
            out_dir,
            filters,
            N_sources,
            variable_source,
            source_positions,
            peak_fluxes,
            i,
            binning_scale,
            circular_aperture,
            overwrite
            )


def generate_gappy_observations(
    out_dir: str,
    n_images: int = 1000,
    circular_aperture: bool = True,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic observation data for testing and following the tutorials.
    
    Parameters
    ----------
    out_dir : str
        The directory to save the data.
    n_images : int, optional
        The number of images to create, by default 100.
    circular_aperture : bool, optional
        Whether to apply a circular aperture shadow to the images, by default True.
    binning_scale : int, optional
        The binning scale of the images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    # create directory if it does not exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    rng = np.random.default_rng(123)
    filters = ["g", "r", "i"]
    
    N_sources = 6
    source_positions = rng.uniform(0 + int(64 / binning_scale), int(2048 / binning_scale - 64 / binning_scale),
                                   (N_sources, 2))  # generate random source positions away from the edges
    peak_fluxes = rng.uniform(100, 1000, N_sources)  # generate random peak fluxes
    variable_source = 1  # index of the variable source
    
    print(f'[OPTICAM] variable source is at ({source_positions[variable_source][0]:.0f}, {source_positions[variable_source][1]:.0f})')
    
    gap_probability = .01  # probability of skipping an image
    
    for i in tqdm(range(n_images), desc="Generating observations", bar_format=bar_format):
        
        # randomly skip some images to create gaps
        if rng.random() < gap_probability:
            # if an image is skipped, increase the probability of skipping the next one to create larger gaps
            gap_probability = .95
            continue
        else:
            gap_probability = .01  # reset the probability of skipping the next image
        
        _create_images(
            out_dir,
            filters,
            N_sources,
            variable_source,
            source_positions,
            peak_fluxes,
            i,
            binning_scale,
            circular_aperture,
            overwrite
            )


