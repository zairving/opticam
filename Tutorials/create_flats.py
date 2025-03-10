import numpy as np
from numpy.typing import NDArray
import os
from astropy.io import fits


def create_base_image(i: int, binning_scale: int) -> NDArray:
    
    rng = np.random.default_rng(i)
    
    base_image = np.zeros((int(2048 / binning_scale), int(2048 / binning_scale))) + 100  # create blank image
    noisy_image = base_image + np.sqrt(base_image) * rng.standard_normal(base_image.shape)  # add Poisson noise
    
    return noisy_image

def apply_flat_field(image: NDArray):
    
    # define mask to apply circular aperture
    x_mid, y_mid = image.shape[1] // 2, image.shape[0] // 2
    distance_from_centre = np.sqrt((x_mid - np.arange(image.shape[1]))**2 + (y_mid - np.arange(image.shape[0]))[:, np.newaxis]**2)
    radius = image.shape[0] // 2
    mask = distance_from_centre >= radius
    
    # create circular aperture shadow
    falloff = 1 / (distance_from_centre[mask] / radius)**2
    
    # apply circular aperture shadow
    image[mask] *= falloff
    
    return image

def create_flats(filters: list, i: int, binning_scale: int, overwrite: bool):
    
    for fltr in filters:
        
        if os.path.isfile(f"Flats/{fltr}-band_image_{i}.fits") and not overwrite:
            continue
        
        noisy_image = create_base_image(i, binning_scale)
        noisy_image = apply_flat_field(noisy_image)  # apply circular aperture shadow
        
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
            hdu.writeto(f"Flats/{fltr}-band_flat_{i}.fits", overwrite=overwrite)
        except:
            pass

def main(overwrite: bool = False):
    
    # create directory if it does not exist
    if not os.path.isdir('Flats/'):
        os.mkdir('Flats/')
    
    filters = ["g", "r", "i"]
    binning_scale = 8
    N_images = 5
    
    for i in range(N_images):
        create_flats(filters, i, binning_scale, overwrite)

if __name__ == "__main__":
    main()