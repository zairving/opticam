from astropy.io import fits
import os
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from typing import List

from create_flats import apply_flat_field


def add_two_dimensional_gaussian_to_image(image: NDArray, x_centroid: float, y_centroid: float, peak_flux: float,
                                          sigma_x: float, sigma_y: float, theta: float) -> NDArray:
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


def variable_function(i: float) -> float:
    """
    Create a variable flux.
    
    Parameters
    ----------
    i : float
        The time.
    
    Returns
    -------
    float
        The flux.
    """
    
    return 5 * np.sin(2 * np.pi * i / 5.5)


def create_base_image(i: int, binning_scale: int) -> NDArray:
    
    rng = np.random.default_rng(i)
    
    base_image = np.zeros((int(2048 / binning_scale), int(2048 / binning_scale))) + 100  # create blank image
    noisy_image = base_image + np.sqrt(base_image) * rng.standard_normal(base_image.shape)  # add Poisson noise
    
    return noisy_image


def create_images(filters: List[str], N_sources: int, variable_source: int, source_positions: NDArray,
                 peak_fluxes: NDArray, i: int, binning_scale: int, overwrite: bool) -> None:
    """
    Create an image for each filter.
    
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
        The time.
    binning_scale : int
        The binning scale.
    overwrite : bool
        Whether to overwrite the image if it already exists.
    """
    
    for fltr in filters:
        
        if os.path.isfile(f"Data/{fltr}-band_image_{i}.fits") and not overwrite:
            continue
        
        noisy_image = create_base_image(i, binning_scale)
        noisy_image = apply_flat_field(noisy_image)  # apply circular aperture shadow
        
        # put sources in the image
        for j in range(N_sources):
            
            if j == variable_source:
                noisy_image = add_two_dimensional_gaussian_to_image(noisy_image, *source_positions[j], peak_fluxes[j] + variable_function(i), 1, 1, 0)
            else:
                noisy_image = add_two_dimensional_gaussian_to_image(noisy_image, *source_positions[j], peak_fluxes[j], 1, 1, 0)
        
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
            hdu.writeto(f"Data/{fltr}-band_image_{i}.fits", overwrite=overwrite)
        except:
            pass


def main(overwrite: bool = False):
    """
    Create test data.
    
    Parameters
    ----------
    overwrite : bool, optional
        Whether to overwrite test data if they currently exist, by default False.
    """
    
    # create directory if it does not exist
    if not os.path.isdir('Data/'):
        os.mkdir('Data/')
    
    rng = np.random.default_rng(123)
    filters = ["g", "r", "i"]
    
    binning_scale = 8
    
    N_sources = 6
    source_positions = rng.uniform(0 + int(64 / binning_scale), int(2048 / binning_scale - 64 / binning_scale), (N_sources, 2))  # generate random source positions away from the edges
    peak_fluxes = rng.uniform(100, 1000, N_sources)  # generate random peak fluxes
    variable_source = 1
    N_images = 100
    
    print('Creating test data...')
    
    for i in tqdm(range(N_images)):
        create_images(filters, N_sources, variable_source, source_positions, peak_fluxes, i, binning_scale, overwrite)

if __name__ == "__main__":
    main()