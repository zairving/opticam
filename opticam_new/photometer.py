from multiprocessing import cpu_count
from typing import List, Literal, Union, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


class Photometer:
    
    def __init__(self, data_directory: str, out_directory: str, date_key: str = "UT",
                 number_of_processors: int = cpu_count(), show_plots: bool = True):
        
        self.data_directory = data_directory
        if self.data_directory[-1] != "/":
            self.data_directory += "/"
        
        if not os.path.isdir(self.data_directory):
            raise FileNotFoundError(f"[OPTICAM] {self.data_directory} not found.")
        
        self.out_directory = out_directory
        if self.out_directory[-1] != "/":
            self.out_directory += "/"
        
        if not os.path.isdir(self.out_directory):
            raise FileNotFoundError(f"[OPTICAM] {self.out_directory} not found. Make sure to run Reducer first so that out_directory contains the necessary files for photometry.")
        
        self.date_key = date_key
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
    
    def get_relative_flux(self, target: List[int], comparison: List[int], flux_type: Literal["aperture", "annulus"],
                          save: bool = True, label: str = None) -> Union[Tuple[ArrayLike, ArrayLike, ArrayLike], None]:
        """
        Compute the relative flux between a target and comparison source for each camera.

        Parameters
        ----------
        target : List[int]
            The catalog IDs of the target source (i.e., [camera 1 ID, camera 2 ID, camera 3 ID]).
        comparison : List[int]
            The catalog IDs of the comparison source (i.e., [camera 1 ID, camera 2 ID, camera 3 ID]).
        save : bool, optional
            Whether to save the resulting light curves to out_directory, by default True.
        label : str, optional
            The label to add to the file name is save is True, by default None.

        Returns
        -------
        Union[Tuple[ArrayLike, ArrayLike, ArrayLike], None]
            If save is False, the time, relative flux, and relative flux error arrays are returned. Otherwise,
            nothing is returned.
        """
        
        if save:
            assert label is not None, "A label must be provided if save is True."

        if flux_type == "aperture":
            save_label = "aperture_light_curve"
            light_curve_dir = "aperture_light_curves"
        elif flux_type == "annulus":
            save_label = "annulus_light_curve"
            light_curve_dir = "annulus_light_curves"
        else:
            print(f"[OPTICAM] Flux type {flux_type} is not supported.")
            return None
        
        t, f, ferr = [], [], []
        
        fig, ax = plt.subplots(nrows=3, tight_layout=True, sharex=True, figsize=(1.5*6.4, 2*4.8))
        
        # for each camera
        for camera in [1, 2, 3]:
            try:
                source_df = pd.read_csv(self.out_directory + f"{light_curve_dir}/camera_{camera}_source_{target[camera - 1]}.csv")
                comparison_df = pd.read_csv(self.out_directory + f"{light_curve_dir}/camera_{camera}_source_{comparison[camera - 1]}.csv")
            except:
                print(f"[OPTICAM] No raw flux files found for camera {camera}, skipping ...")
                print(self.out_directory + f"{light_curve_dir}/camera_{camera}_source_{target[camera - 1]}.csv")
                print(f"{light_curve_dir}/camera_{camera}_source_{comparison[camera - 1]}.csv")
                continue
            
            # ensure time arrays match, otherwise relative flux cannot be calculated
            if not np.array_equal(source_df["MJD"].values, comparison_df["MJD"].values):
                print(f"[OPTICAM] Time arrays for camera {camera} do not match, skipping ...")
                continue
            
            relative_flux = source_df["flux"].values/comparison_df["flux"].values
            relative_flux_error = relative_flux*np.sqrt(np.square(source_df["flux_error"].values/source_df["flux"].values) + np.square(comparison_df["flux_error"].values/comparison_df["flux"].values))
            
            t.append(source_df["MJD"].values)
            f.append(relative_flux)
            ferr.append(np.abs(relative_flux_error))
            
            ax[camera - 1].errorbar(source_df["MJD"].values, relative_flux, np.abs(relative_flux_error), fmt="k.", ms=2, ecolor="grey", elinewidth=1)
            ax[camera - 1].set_title(f"Camera {camera} (Source ID: {target[camera - 1]}, Comparison ID: {comparison[camera - 1]})")
        
        ax[2].set_xlabel("MJD")
        ax[1].set_ylabel("Relative flux")

        fig.savefig(self.out_directory + f"{label}_{save_label}.png")

        if self.show_plots:
            plt.show()
        
        if save:
            # for each camera
            for i in range(len(t)):
                # create light curve dictionary
                light_curve = {"MJD": t[i], "flux": f[i], "flux_error": ferr[i]}
                self._save_light_curve(light_curve, f"camera_{i + 1}_{label}_{save_label}.txt", target=target[i], comparison=comparison[i])
        else:
            return t, f, ferr
    
    def _save_light_curve(self, light_curve: dict, file_name: str, target: int, comparison: int) -> None:
        """
        Saves the light curve to out_directory under file_name.

        Parameters
        ----------
        light_curve : dict
            The light curve to be saved.
        file_name : str
            The name of the file.
        target : int
            The target source's catalog ID.
        comparison : int
            The comparison source's catalog ID.
        """
        
        header = list(light_curve.keys())

        with open(self.out_directory + file_name, "w") as file:
            file.write(f"# target: {target}\n")
            file.write(f"# comparison: {comparison}\n")
            
            for key in header:
                file.write(f"{key} ")
            file.write("\n")
            
            for i in range(len(light_curve[header[0]])):
                for key in header:
                    file.write(f"{light_curve[key][i]} ")
                file.write("\n")

    def get_relative_error_light_curve(self, target: str, flux_type_1: str, flux_type_2: str) -> None:
        
        fig, ax = plt.subplots(nrows=3, tight_layout=True, sharex=True, figsize=(6.4, 2*4.8))
        
        for camera in [1, 2, 3]:
            try:
                df1 = pd.read_csv(self.out_directory + f"camera_{camera}_{target}_{flux_type_1}_light_curve.txt", delimiter=" ", comment="#")
            except:
                print(f"[OPTICAM] {self.out_directory}camera_{camera}_{target}_{flux_type_1}_light_curve.txt could not be loaded, skipping ...")
                continue
            
            try:
                df2 = pd.read_csv(self.out_directory + f"camera_{camera}_{target}_{flux_type_2}_light_curve.txt", delimiter=" ", comment="#")
            except:
                print(f"[OPTICAM] {self.out_directory}camera_{camera}_{target}_{flux_type_2}_light_curve.txt could not be loaded, skipping ...")
                continue
            
            assert np.array_equal(df1["MJD"].values, df2["MJD"].values), f"Time arrays for camera {camera} do not match."
            
            f = (df1["flux"].values - df2["flux"].values)/df1["flux"].values
            temp = np.sqrt(np.square(df1["flux_error"].values/df1["flux"].values) + np.square(df2["flux_error"].values/df2["flux"].values))
            temp2 = np.sqrt(np.square(temp/(df1["flux"].values - df2["flux"].values)) + np.square(df1["flux_error"].values/df1["flux"].values))
            ferr = np.abs(f)*temp2
            
            ax[camera - 1].errorbar(df1["MJD"], f, ferr, fmt="k.", ms=2, ecolor="grey", elinewidth=1)
            
        ax[2].set_xlabel("MJD")
        ax[1].set_ylabel(r"$(F_{\rm " + flux_type_1.replace("_", "\\_") + r"} - F_{\rm " + flux_type_2.replace("_", "\\_") + r"})/F_{\rm " + flux_type_1.replace("_", "\\_") + r"}$")
        
        fig.savefig(self.out_directory + f"{target}_relative_error_light_curve.png")