from astropy.io import fits
import os
from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike


def add_two_dimensional_gaussian_to_image(image: ArrayLike, x_centroid: float, y_centroid: float, peak_flux: float,
                                          sigma_x: float, sigma_y: float, theta: float) -> ArrayLike:
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
    b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
    c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)
    
    gaussian = peak_flux*np.exp(-(a*(x - x_centroid)**2 + 2*b*(x - x_centroid)*(y - y_centroid) + c*(y - y_centroid)**2))
    
    mask = gaussian > gaussian.max()/100
    source = np.where(mask, gaussian, 0)
    
    return image + gaussian

def variable_function(i: float) -> float:
    
    return 10*np.sin(2*np.pi*i/50)

save_path = input("Enter the path to save the fits file: ")

if not save_path.endswith("/"):
    save_path += "/"

# create directory if it does not exist
if not os.path.isdir(save_path):
    directories = save_path.split("/")[1:-1]  # remove leading and trailing slashes
    for i in range(len(directories)):
        if not os.path.isdir("/" + "/".join(directories[:i + 1])):
            try:
                os.mkdir("/" + "/".join(directories[:i + 1]))
            except:
                raise FileNotFoundError(f"Could not create directory {directories[:i + 1]}")

rng = np.random.default_rng(123)
filters = ["g", "r", "i"]

N_sources = 6
source_positions = rng.uniform(17, 239, (N_sources, 2))  # generate random source positions
peak_fluxes = rng.uniform(10000, 100000, N_sources)  # generate random peak fluxes
variable_source = 1

N_images = 1000

for i in tqdm(range(N_images)):
    
    base_image = np.zeros((256, 256)) + 1000  # create blank image
    noisy_image = base_image + np.sqrt(base_image)*rng.standard_normal(base_image.shape)  # add Poisson noise
    
    # add background gradient
    ny, nx = noisy_image.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x*y/100
    noisy_image = noisy_image + gradient
    
    for fltr in filters:
        
        indx = filters.index(fltr)
        
        filter_image = noisy_image + i
        
        # put sources in the image
        for j in range(N_sources):
            
            if j == variable_source:
                filter_image = add_two_dimensional_gaussian_to_image(filter_image, *source_positions[j], peak_fluxes[j]+variable_function(i), 1, 1, 0)
            else:
                filter_image = add_two_dimensional_gaussian_to_image(filter_image, *source_positions[j], peak_fluxes[j], 1, 1, 0)
        
        # create fits file
        hdu = fits.PrimaryHDU(filter_image)
        hdu.header["FILTER"] = fltr
        hdu.header["BINNING"] = "8x8"
        hdu.header["GAIN"] = 1.
        
        # create observation time
        hh = str(i // 3600).zfill(2)
        mm = str(i % 3600 // 60).zfill(2)
        ss = str(i % 60).zfill(2)
        hdu.header["UT"] = f"2024-01-01 {hh}:{mm}:{ss}"
        
        # save fits file
        hdu.writeto(f"{save_path}/{fltr}-band_image_{i}.fits", overwrite=True)
