from multiprocessing import cpu_count
from typing import List, Literal, Union, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


class Photometer:
    
    def __init__(self, out_directory: str, show_plots: bool = True):
        
        self.out_directory = out_directory
        if self.out_directory[-1] != "/":
            self.out_directory += "/"
        
        if not os.path.isdir(self.out_directory):
            raise FileNotFoundError(f"[OPTICAM] {self.out_directory} not found. Make sure to run Reducer first so that out_directory contains the necessary files for photometry.")
        
        self.show_plots = show_plots
        
        # read filters
        self.filters = []
        with open(self.out_directory + "misc/filters.txt", "r") as file:
            for line in file:
                self.filters.append(line.strip())
        print(self.filters)
    
    def get_relative_flux(self, target: List[int], comparison: List[Union[List, int]], phot_type: Literal["aperture", "annulus", "normal", "optimal"],
                          save: bool = True, label: str = None) -> Union[Tuple[ArrayLike, ArrayLike, ArrayLike], None]:
        
        if save:
            assert label is not None, "A label must be provided if save is True."
        
        # ensure comparison is a list of lists
        for i in range(len(comparison)):
            if isinstance(comparison[i], int):
                comparison[i] = [comparison[i]]

        if phot_type == "aperture":
            save_label = "aperture_light_curve"
            light_curve_dir = "aperture_light_curves"
        elif phot_type == "annulus":
            save_label = "annulus_light_curve"
            light_curve_dir = "annulus_light_curves"
        elif phot_type == "normal":
            save_label = "normal_light_curve"
            light_curve_dir = "normal_light_curves"
        else:
            print(f"[OPTICAM] Flux type {phot_type} is not supported.")
            return None
        
        mjd, t, f, ferr, flags = [], [], [], [], []
        
        ref = np.loadtxt(self.out_directory + "misc/earliest_observation_time.txt")
        
        fig, ax = plt.subplots(nrows=3, tight_layout=True, sharex=True, figsize=(1.5*6.4, 2*4.8))
        
        # for each camera
        for fltr in self.filters:
            try:
                source_df = pd.read_csv(self.out_directory + f"{light_curve_dir}/{fltr}_source_{target[self.filters.index(fltr)]}.csv")
            except:
                print(f"[OPTICAM] Could not load {light_curve_dir}/{fltr}_source_{target[self.filters.index(fltr)]}.csv, skipping ...")
                continue
            
            transformed_mask = source_df["quality_flag"] == "A"
            
            time = source_df["MJD"].values.copy()
            time -= ref
            time *= 86400
            
            comp_fluxes = np.zeros_like(source_df["flux"].values)
            comp_flux_errors = np.zeros_like(source_df["flux_error"].values)
            
            # for each comparison source
            for comp in comparison[self.filters.index(fltr)]:
                # load comparison source light curve
                try:
                    comparison_df = pd.read_csv(self.out_directory + f"{light_curve_dir}/{fltr}_source_{comp}.csv")
                except:
                    print(f"[OPTICAM] Could not load {light_curve_dir}/{fltr}_source_{comparison[self.filters.index(fltr)]}.csv, skipping ...")
                    continue
                
                # ensure time arrays match, otherwise relative flux cannot be calculated
                if not np.array_equal(source_df["MJD"].values, comparison_df["MJD"].values):
                    print(f"[OPTICAM] {fltr} time arrays do not match, skipping ...")
                    continue
                
                # add fluxes
                comp_fluxes += comparison_df["flux"].values
                
                # add flux errors in quadrature
                comp_flux_errors = np.sqrt(np.square(comp_flux_errors) + np.square(comparison_df["flux_error"].values))
                
            # calculate relative flux and error
            relative_flux = source_df["flux"].values/comp_fluxes
            relative_flux_error = relative_flux*np.sqrt(np.square(source_df["flux_error"].values/source_df["flux"].values) + np.square(comp_flux_errors/comp_fluxes))
            
            mjd.append(source_df["MJD"].values)
            t.append(time)
            f.append(relative_flux)
            ferr.append(np.abs(relative_flux_error))
            flags.append(source_df["quality_flag"].values)
            
            # ax[self.filters.index(fltr)].errorbar(time[transformed_mask], relative_flux[transformed_mask], np.abs(relative_flux_error[transformed_mask]), fmt="k.", ms=2, ecolor="grey", elinewidth=1)
            # ax[self.filters.index(fltr)].errorbar(time[~transformed_mask], relative_flux[~transformed_mask], np.abs(relative_flux_error[~transformed_mask]), fmt="r.", ms=2, elinewidth=1, alpha=.2)
            ax[self.filters.index(fltr)].plot(time[transformed_mask], relative_flux[transformed_mask], "k.", ms=2)
            ax[self.filters.index(fltr)].plot(time[~transformed_mask], relative_flux[~transformed_mask], "r.", ms=2, alpha=.2)
            ax[self.filters.index(fltr)].set_title(f"{fltr} (Source ID: {target[self.filters.index(fltr)]}, Comparison ID(s): {comparison[self.filters.index(fltr)]})")
        
        ax[2].set_xlabel(f"Time from MJD {ref} [s]")
        ax[1].set_ylabel("Relative flux")

        fig.savefig(self.out_directory + f"{label}_{save_label}.png")

        if self.show_plots:
            plt.show()
        
        if save:
            # for each camera
            for i in range(len(t)):
                # create light curve dictionary
                light_curve = {"MJD": mjd[i], "flux": f[i], "flux_error": ferr[i], "quality_flag": flags[i]}
                self._save_light_curve(light_curve, f"{self.filters[i]}_{label}_{save_label}.txt", target=target[i], comparison=comparison[i])
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

    def get_relative_error_light_curve(self, target: str, phot_type_1: str, phot_type_2: str) -> None:
        
        fig, ax = plt.subplots(nrows=3, tight_layout=True, sharex=True, figsize=(6.4, 2*4.8))
        
        for fltr in self.filters:
            try:
                df1 = pd.read_csv(self.out_directory + f"{fltr}_{target}_{phot_type_1}_light_curve.txt", delimiter=" ", comment="#")
            except:
                print(f"[OPTICAM] {self.out_directory}{fltr}_{target}_{phot_type_1}_light_curve.txt could not be loaded, skipping ...")
                continue
            
            try:
                df2 = pd.read_csv(self.out_directory + f"{fltr}_{target}_{phot_type_2}_light_curve.txt", delimiter=" ", comment="#")
            except:
                print(f"[OPTICAM] {self.out_directory}{fltr}_{target}_{phot_type_2}_light_curve.txt could not be loaded, skipping ...")
                continue
            
            assert np.array_equal(df1["MJD"].values, df2["MJD"].values), f"{fltr} time arrays do not match."
            
            f = (df1["flux"].values - df2["flux"].values)/df1["flux"].values
            temp = np.sqrt(np.square(df1["flux_error"].values/df1["flux"].values) + np.square(df2["flux_error"].values/df2["flux"].values))
            temp2 = np.sqrt(np.square(temp/(df1["flux"].values - df2["flux"].values)) + np.square(df1["flux_error"].values/df1["flux"].values))
            ferr = np.abs(f)*temp2
            
            ax[self.filters.index(fltr)].errorbar(df1["MJD"], f, ferr, fmt="k.", ms=2, ecolor="grey", elinewidth=1)
            
        ax[2].set_xlabel("MJD")
        ax[1].set_ylabel(r"$(F_{\rm " + phot_type_1.replace("_", "\\_") + r"} - F_{\rm " + phot_type_2.replace("_", "\\_") + r"})/F_{\rm " + phot_type_1.replace("_", "\\_") + r"}$")
        
        fig.savefig(self.out_directory + f"{target}_{phot_type_1}-{phot_type_2}_relative_error_light_curve.png")