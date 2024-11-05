import os
from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List

from opticam_new.helpers import log_binnings, log_filters

class Corrector:
    """
    Helper class for correcting OPTICAM images.
    """
    
    def __init__(self, out_dir: str, flats_dir: str = None, c1_flats_dir: str = None, c2_flats_dir: str = None,
                 c3_flats_dir: str = None) -> None:
        """
        Helper class for correcting OPTICAM images.

        Parameters
        ----------
        out_dir : str
            The output directory.
        flats_dir : str, optional
            The directory path to the flat-field images, by default None (no flat-field correction). Note: this
            should point to a directory containing only flat-field images, NOT to a master flat-field image(s). The
            master flat-field image(s) will be created from the flat-field images in this directory and saved in
            out_dir/calibration_images. This parameter will override c1_flats_dir, c2_flats_dir, and c3_flats_dir if
            they are also defined. flats_dir is assumed to contain flat-field images for all cameras being used.
        c1_flats_dir : str, optional
            The directory path to the flat-field images for C1, by default None. If flats_dir is also defined, this
            parameter will be ignored.
        c2_flats_dir : str, optional
            The directory path to the flat-field images for C2, by default None. If flats_dir is also defined, this
            parameter will be ignored.
        c3_flats_dir : str, optional
            The directory path to the flat-field images for C3, by default None. If flats_dir is also defined, this
            parameter will be ignored.
        """
        
        self.out_dir = out_dir
        
        if not self.out_dir.endswith("/"):
            self.out_dir += "/"
        
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except:
                raise Exception(f"Could not create output directory {out_dir}").__annotations__
        
        flat_paths = []
        
        if flats_dir is not None:
            if not flats_dir.endswith("/"):
                flats_dir += "/"
            
            if not os.path.exists(flats_dir):
                raise Exception(f"Flat-field images directory {flats_dir} does not exist").__annotations__
            
            flat_paths += [flats_dir + flat for flat in sorted(os.listdir(flats_dir))]
        else:
            if c1_flats_dir is not None:
                if not c1_flats_dir.endswith("/"):
                    c1_flats_dir += "/"
                
                if not os.path.exists(c1_flats_dir):
                    raise Exception(f"Flat-field images directory {c1_flats_dir} does not exist").__annotations__
                
                flat_paths += [c1_flats_dir + flat for flat in sorted(os.listdir(c1_flats_dir))]
            
            if c2_flats_dir is not None:
                if not c2_flats_dir.endswith("/"):
                    c2_flats_dir += "/"
                
                if not os.path.exists(c2_flats_dir):
                    raise Exception(f"Flat-field images directory {c2_flats_dir} does not exist").__annotations__
                
                flat_paths += [c2_flats_dir + flat for flat in sorted(os.listdir(c2_flats_dir))]
            
            if c3_flats_dir is not None:
                if not c3_flats_dir.endswith("/"):
                    c3_flats_dir += "/"
                
                if not os.path.exists(c3_flats_dir):
                    raise Exception(f"Flat-field images directory {c3_flats_dir} does not exist").__annotations__
                
                flat_paths += [c3_flats_dir + flat for flat in sorted(os.listdir(c3_flats_dir))]
        
        # get flats for each filter
        self.flat_paths = self._validate_flat_files(sorted(flat_paths))
    
    
    def _validate_flat_files(self, flat_paths: List[str]) -> Dict[str, List[str]]:
        """
        Ensure that the flat-field images in the specified directory are valid (i.e., contain at most three filters
        and use the same binning).
        
        Parameters
        ----------
        flat_paths : List[str]
            The paths to the flat-field images.
        
        Returns
        -------
        Dict[str, List[str]]
            A dictionary containing the paths to the flat-field images for each filter.
        """
        
        filters, binnings = {}, {}
        
        for flat_path in flat_paths:
            with fits.open(flat_path) as hdul:
                header = hdul[0].header
                filters[flat_path] = (header["FILTER"])
                binnings[flat_path] = (header["BINNING"])
        
        unique_filters = set(filters.values())
        unique_binnings = set(binnings.values())
        
        if len(unique_filters) > 3:
            log_filters(flat_paths, self.out_dir)
            raise ValueError(f'[OPTICAM] More than three filters found in the flat-field images. Image filters have been logged to {self.out_dir}misc/filters.json')
        
        if len(unique_binnings) > 1:
            log_binnings(flat_paths, self.out_dir)
            raise ValueError(f'[OPTICAM] Inconsistent binning detected in the flat-field images. Image binnings have been logged to {self.out_dir}misc/binnings.json')
        
        # get flats for each filter
        flats = {}
        for fltr in unique_filters:
            flats[fltr + '-band'] = []
            for (k, v) in filters.items():
                if v == fltr:
                    flats[fltr + '-band'].append(k)
        
        return flats
    
    def create_master_flats(self, overwrite: bool = False) -> None:
        """
        Create master flat-field images for each available filter.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the existing master flat-field image, by default False.
        """
        
        for fltr in self.flat_paths.keys():
            # skip if master flat-field image already exists and overwrite is False
            if os.path.exists(self.out_dir + f"corr/{fltr}_master_flat.fit.gz") and not overwrite:
                return
            
            if not os.path.isdir(self.out_dir + "corr/"):
                try:
                    os.makedirs(self.out_dir + "corr/")
                except:
                    raise Exception(f"[OPTICAM] Could not create corr directory in {self.out_dir}").__annotations__
            
            if len(self.flat_paths[fltr]) == 1:
                raise Exception(f"[OPTICAM] Only one {fltr} flat found. Master flats cannot be created from a single image.").__annotations__
            
            # read flats
            flats = []
            for flat_path in self.flat_paths[fltr]:
                with fits.open(flat_path) as hdul:
                    flat = np.array(hdul[0].data)
                    flats.append(flat / np.median(flat))  # add normalised flat-field image to list
            
            # create master flat
            master_flat = np.median(flats, axis=0)
            
            # save master flat to file
            hdu = fits.PrimaryHDU(master_flat)
            hdu.writeto(self.out_dir + f"corr/{fltr}_master_flat.fit.gz", overwrite=overwrite)
    
    def flat_correct(self, image: NDArray, fltr: str) -> NDArray:
        """
        Correct an image for flat-fielding.
        
        Parameters
        ----------
        image : np.ndarray
            The image to correct.
        fltr : str
            The image filter.
        
        Returns
        -------
        NDArray
            The corrected image.
        """
        
        if fltr not in self.flat_paths.keys():
            raise ValueError(f"[OPTICAM] No flat-field images found for {fltr} filter.")
        
        if not os.path.exists(self.out_dir + f"corr/{fltr}_master_flat.fit.gz") and fltr in self.flat_paths.keys():
            print(f"[OPTICAM] {fltr} master flat-field image not found. Attempting to create...")
            try:
                self.create_master_flat()
                print("[OPTICAM] Master flat-field image created.")
            except:
                raise Exception("[OPTICAM] Could not create master flat-field image(s).").__annotations__
        
        # load master flat
        with fits.open(self.out_dir + f"corr/{fltr}_master_flat.fit.gz") as hdul:
            master_flat = np.array(hdul[0].data)
        
        # correct image for flat-fielding
        return image / master_flat








