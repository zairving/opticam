from astropy.io import fits
import os
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray


def add_two_dimensional_gaussian_to_image(image: NDArray, x_centroid: float, y_centroid: float, peak_flux: float,
                                          sigma_x: float, sigma_y: float, theta: float) -> NDArray:
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    
    gaussian = peak_flux*np.exp(-(a*(x - x_centroid)**2 + 2*b*(x - x_centroid)*(y - y_centroid) + c*(y - y_centroid)**2))
    
    return image + gaussian


def variable_function(i: float) -> float:
    
    return 5 * np.sin(2 * np.pi * i / 5.5)


def create_image(filters, N_sources, variable_source, source_positions, peak_fluxes, i, binning_scale, overwrite):
    
    if os.path.isfile(f"Data/{filters[0]}-band_image_{i}.fits") and not overwrite:
        return
    
    rng = np.random.default_rng(i)
    
    base_image = np.zeros((int(2048 / binning_scale), int(2048 / binning_scale))) + 100  # create blank image
    noisy_image = base_image + np.sqrt(base_image) * rng.standard_normal(base_image.shape)  # add Poisson noise
    
    # add background gradient
    ny, nx = noisy_image.shape
    y, x = np.mgrid[:ny, :nx]
    noisy_image = noisy_image
    
    for fltr in filters:
        
        filter_image = noisy_image
        
        # put sources in the image
        for j in range(N_sources):
            
            if j == variable_source:
                filter_image = add_two_dimensional_gaussian_to_image(filter_image, *source_positions[j], peak_fluxes[j] + variable_function(i), 1, 1, 0)
            else:
                filter_image = add_two_dimensional_gaussian_to_image(filter_image, *source_positions[j], peak_fluxes[j], 1, 1, 0)
        
        # create fits file
        hdu = fits.PrimaryHDU(filter_image)
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
        create_image(filters, N_sources, variable_source, source_positions, peak_fluxes, i, binning_scale, overwrite)

if __name__ == "__main__":
    main()