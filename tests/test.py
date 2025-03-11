import unittest
import numpy as np
from photutils.segmentation import SourceCatalog
import os
import tempfile
from astropy.io import fits

from opticam_new.background import Background
from opticam_new.local_background import CircularLocalBackground, EllipticalLocalBackground
from opticam_new.finder import CrowdedFinder, Finder
from opticam_new.generate import create_synthetic_observations, create_synthetic_flats


class TestBackground(unittest.TestCase):
    
    def test_background(self):
        """
        Test the default background estimator.
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            dir_path = os.path.join(temp_dir, 'observations')
            
            if not os.path.isdir(dir_path):
                create_synthetic_observations(dir_path, n_images=5, circular_aperture=False)
            
            bkg_estimator = Background()
            
            for im in os.listdir(dir_path):
                    with fits.open(os.path.join(dir_path, im)) as hdul:
                        image = np.array(hdul[0].data)
                    
                    bkg = bkg_estimator(image)
                    
                    self.assertTrue(np.allclose(bkg.background, 100, rtol = .05))
                    self.assertTrue(np.allclose(bkg.background_rms, 10, rtol = .1))


class TestFinder(unittest.TestCase):
    
    def test_finder(self):
        """
        Test the source finder.
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            dir_path = os.path.join(temp_dir, 'observations')
            
            if not os.path.isdir(dir_path):
                create_synthetic_observations(dir_path, n_images=5, circular_aperture=False)
            
            bkg_estimator = Background()
            finder = Finder()
            
            
            for im in os.listdir(dir_path):
                with fits.open(os.path.join(dir_path, im)) as hdul:
                    image = np.array(hdul[0].data)
                
                bkg = bkg_estimator(image)
                segment_map = finder(image - bkg.background, 5 * bkg.background_rms)
                
                self.assertTrue(segment_map.nlabels == 6)


class TestCrowdedFinder(unittest.TestCase):
    
    def test_finder(self):
        """
        Test the crowded source finder.
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            dir_path = os.path.join(temp_dir, 'observations')
            
            if not os.path.isdir(dir_path):
                create_synthetic_observations(dir_path, n_images=5, circular_aperture=False)
            
            bkg_estimator = Background()
            finder = CrowdedFinder()
            
            for im in os.listdir(dir_path):
                with fits.open(os.path.join(dir_path, im)) as hdul:
                    image = np.array(hdul[0].data)
                
                bkg = bkg_estimator(image)
                segment_map = finder(image - bkg.background, 5 * bkg.background_rms)
                
                self.assertTrue(segment_map.nlabels == 6)


class TestCircularLocalBackground(unittest.TestCase):
    
    def test_circular_local_background(self):
        """
        Test the local background estimator.
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            dir_path = os.path.join(temp_dir, 'observations')
            
            if not os.path.isdir(dir_path):
                create_synthetic_observations(dir_path, n_images=5, circular_aperture=False)
            
            bkg_estimator = Background()
            finder = Finder()
            
            for im in os.listdir(dir_path):
                with fits.open(os.path.join(dir_path, im)) as hdul:
                    image = np.array(hdul[0].data)
                
                bkg = bkg_estimator(image)
                segment_map = finder(image - bkg.background, 5 * bkg.background_rms)
                tbl = SourceCatalog(image - bkg.background, segment_map, background=bkg.background).to_table()
                coords = np.array([tbl["xcentroid"], tbl["ycentroid"]]).T
                local_bkg_estimator = CircularLocalBackground()
                
                for i in range(len(coords)):
                    local_bkg, local_bkg_error = local_bkg_estimator(image, np.sqrt(image),
                                                                     5 * tbl['semimajor_sigma'][i].value, 
                                                                     5 * tbl['semiminor_sigma'][i].value,
                                                                     tbl['orientation'][i], coords[i])
                    
                    self.assertTrue(np.allclose(local_bkg, 100, rtol = 0.05))
                    self.assertTrue(np.allclose(local_bkg_error, 10, rtol = 0.2))



class TestEllipticalLocalBackground(unittest.TestCase):
    
    def test_elliptical_local_background(self):
        """
        Test the local background estimator.
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            
            dir_path = os.path.join(temp_dir, 'observations')
            
            if not os.path.isdir(dir_path):
                create_synthetic_observations(dir_path, n_images=5, circular_aperture=False)
            
            bkg_estimator = Background()
            finder = Finder()
            
            for im in os.listdir(dir_path):
                with fits.open(os.path.join(dir_path, im)) as hdul:
                    image = np.array(hdul[0].data)
                
                bkg = bkg_estimator(image)
                segment_map = finder(image - bkg.background, 5 * bkg.background_rms)
                tbl = SourceCatalog(image - bkg.background, segment_map, background=bkg.background).to_table()
                coords = np.array([tbl["xcentroid"], tbl["ycentroid"]]).T
                local_bkg_estimator = EllipticalLocalBackground()
                
                for i in range(len(coords)):
                    local_bkg, local_bkg_error = local_bkg_estimator(image, np.sqrt(image),
                                                                     5 * tbl['semimajor_sigma'][i].value, 
                                                                     5 * tbl['semiminor_sigma'][i].value,
                                                                     tbl['orientation'][i], coords[i])
                    
                    self.assertTrue(np.allclose(local_bkg, 100, rtol = 0.05))
                    self.assertTrue(np.allclose(local_bkg_error, 10, rtol = 0.2))