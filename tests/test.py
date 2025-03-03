import unittest
import numpy as np
from numpy.typing import NDArray

from opticam_new.background import Background
from opticam_new.finder import CrowdedFinder, Finder


def add_two_dimensional_gaussian_to_image(image: NDArray, x_centroid: float, y_centroid: float, peak_flux: float,
                                          sigma_x: float, sigma_y: float, theta: float) -> NDArray:
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = - np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    
    gaussian = peak_flux * np.exp(- (a * (x - x_centroid)**2 + 2 * b * (x - x_centroid) * (y - y_centroid) + 
                                     c * (y - y_centroid)**2))
    
    return image + gaussian

def generate_image():
    rng = np.random.default_rng(0)
    
    base_image = np.zeros((2048, 2048)) + 100  # create blank image
    noisy_image = base_image + np.sqrt(base_image) * rng.standard_normal(base_image.shape)  # add Poisson noise
    
    for i in range(3):
        
        # prevent source being positioned near edge of image
        x = rng.uniform(128, 2048 - 128)
        y = rng.uniform(128, 2048 - 128)
        
        noisy_image = add_two_dimensional_gaussian_to_image(noisy_image, x, y, 1000, 10, 10, 0)
    
    return noisy_image


class TestBackground(unittest.TestCase):
    
    def test_background(self):
        """
        Test the default background estimator.
        """
        
        image = generate_image()
        
        bkg_estimator = Background()
        bkg = bkg_estimator(image)
        
        self.assertTrue(np.allclose(bkg.background, 100, rtol=.01))  # check background is within 1% of true value
        self.assertTrue(np.allclose(bkg.background_rms, 10, rtol=.05))  # check background RMS is within 5% of true value


class TestFinder(unittest.TestCase):
    
    def test_finder(self):
        """
        Test the source finder.
        """
        
        image = generate_image()
        
        bkg_estimator = Background()
        bkg = bkg_estimator(image)
        
        finder = Finder()
        segment_map = finder(image - bkg.background, 5 * bkg.background_rms)
        
        self.assertTrue(segment_map.nlabels == 3)  # check that all three sources were found


class TestCrowdedFinder(unittest.TestCase):
    
    def test_finder(self):
        """
        Test the crowded source finder.
        """
        
        image = generate_image()
        
        bkg_estimator = Background()
        bkg = bkg_estimator(image)
        
        finder = CrowdedFinder()
        segment_map = finder(image - bkg.background, 5 * bkg.background_rms)
        
        self.assertTrue(segment_map.nlabels == 3)  # check that all three sources were found