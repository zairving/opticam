from typing import Dict, List, Literal, Union, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from astropy.table import QTable
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from matplotlib import image as mpimage
import json

from opticam_new.analyser import Analyser


class Photometer:
    """
    Helper class for creating light curves from reduced OPTICam data.
    """
    
    def __init__(self, out_directory: str, show_plots: bool = True):
        """
        Helper class for creating light curves from reduced OPTICam data.
        
        Parameters
        ----------
        out_directory : str
            The path to the directory where output will be saved.
        show_plots : bool, optional
            Whether plots should be shown as they're generated, by default True.
        
        Raises
        ------
        FileNotFoundError
            If out_directory cannot be found.
        """
        
        self.out_directory = out_directory
        if self.out_directory[-1] != "/":
            self.out_directory += "/"
        
        if not os.path.isdir(self.out_directory):
            raise FileNotFoundError('[OPTICAM] ' + self.out_directory + ' not found. Make sure to run Reducer first so that out_directory contains the necessary files for photometry.')
        
        self.show_plots = show_plots
        
        with open(os.path.join(self.out_directory, 'misc/input_parameters.json'), 'r') as file:
            input_parameters = json.load(file)
        self.filters = input_parameters['filters']
        self.t_ref = input_parameters['t_ref']
        print(self.filters)
        
        # read catalogs
        self.catalogs = {}
        for fltr in self.filters:
            try:
                self.catalogs.update({f"{fltr}": QTable.read(self.out_directory + f"cat/{fltr}_catalog.ecsv", format="ascii.ecsv")})
            except:
                print(f"[OPTICAM] Could not load {self.out_directory}cat/{fltr}_catalog.ecsv, skipping ...")
                self.filters.remove(fltr)
                continue
        
        # plot catalogs
        catalog_image = mpimage.imread(self.out_directory + "cat/catalogs.png")
        fig, ax = plt.subplots(figsize=(len(self.catalogs)*5, 5))
        ax.imshow(catalog_image)
        
        # remove ticks and tick labels
        ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        
        plt.show()
    
    def get_relative_light_curve(self, fltr: str, target: int, comparisons: List[int],
                                 phot_type: Literal["aperture", "annulus", "normal", "optimal"],
                                 prefix: str = None, match_other_cameras = False,
                                 show_diagnostics: bool = True) -> Analyser:
        """
        Compute the relative light curve for a target source with respect to one or more comparison sources. By default,
        the relative light curve is computed for a single filter. The relative light curve is saved to
        out_directory/relative_light_curves. To automatically match the target and comparison sources across the other two
        filters, set match_other_cameras to True. Note that this can incorrectly match sources, so it is recommended to
        manually check the results.
        
        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve for.
        target : int
            The catalog ID of the target source.
        comparison : List[int]
            The catalog ID(s) of the comparison source(s).
        phot_type : Literal['aperture', 'annulus', 'normal', 'optimal']
            The type of photometry to use.
        prefix : str, optional
            The prefix to use when saving the relative light curve (e.g., the target star's name), by default None. 
        match_other_cameras : bool, optional
            Whether to try and automatically match the target and comparison sources across OPTICAM's other cameras, by
            default False. Note that this can incorrectly match sources, particularly if the fields are crowded, and so
            it is recommended to manually check the results.
        show_diagnostics : bool, optional
            Whether to show diagnostic plots for each comparison source, by default True.
        
        Returns
        -------
        Analyser
            An Analyser object containing the relative light curve(s).
        """
        
        if phot_type == "aperture":
            save_label = "aperture_light_curve"
            light_curve_dir = "aperture_light_curves"
        elif phot_type == "annulus":
            save_label = "annulus_light_curve"
            light_curve_dir = "annulus_light_curves"
        elif phot_type == "normal":
            save_label = "normal_light_curve"
            light_curve_dir = "normal_light_curves"
        elif phot_type == "optimal":
            save_label = "optimal_light_curve"
            light_curve_dir = "optimal_light_curves"
        else:
            print(f"[OPTICAM] Flux type {phot_type} is not supported.")
            return None
        
        if not os.path.isdir(self.out_directory + "relative_light_curves"):
            os.mkdir(self.out_directory + "relative_light_curves")
        
        # validate filter
        if fltr not in self.filters:
            if fltr not in [cat_filter[0] for cat_filter in self.filters]:
                raise ValueError('[OPTICAM] ' + fltr + ' is not a valid filter.')
        
        if not match_other_cameras:
            # compute and plot relative light curve for single filter
            relative_light_curve, transformed_mask = self._compute_relative_light_curve(fltr, target, comparisons, phot_type, self.t_ref, show_diagnostics)
            self._plot_relative_light_curve(relative_light_curve, self.t_ref, transformed_mask, target=target, comparisons=comparisons, prefix=prefix, fltr=fltr, save_label=save_label)
            
            # save light curve to CSV
            relative_light_curve.to_csv(self.out_directory + "relative_light_curves/" + f"{prefix}_{fltr}_{save_label}.csv", index=False)
            
            # return Analyser object
            return Analyser({fltr: relative_light_curve}, self.out_directory, prefix, phot_type)
        else:
            # define dictionaries to store relative light curves and transformed masks for each camera
            relative_light_curves = {}
            transformed_masks = {}
            targets_ = {}
            comparisons_ = {}
            
            # get source coordinates for input filter
            input_filter_coords = np.array([self.catalogs[fltr]['xcentroid'], self.catalogs[fltr]['ycentroid']]).T
            
            for cat_fltr in self.filters:
                # get target and comparison source indices
                if cat_fltr == fltr:
                    # if the current filter is the input filter, the target and comparison sources are already known
                    targets_[cat_fltr] = target
                    comparisons_[cat_fltr] = comparisons
                else:
                    # if the current filter is not the input filter, the target and comparison sources need to be matched
                    # using the Hungarian algorithm
                    fltr_coords = np.array([self.catalogs[cat_fltr]['xcentroid'], self.catalogs[cat_fltr]['ycentroid']]).T  # get source coordinates for current filter
                    distance_matrix = cdist(input_filter_coords, fltr_coords)  # compute distance matrix
                    input_filter_indices, fltr_indices = linear_sum_assignment(distance_matrix)  # solve assignment problem
                    
                    # get target and comparison source indices
                    targets_[cat_fltr] = int(fltr_indices[np.where(input_filter_indices == target - 1)[0]]) + 1
                    comparisons_[cat_fltr] = [int(fltr_indices[np.where(input_filter_indices == comp - 1)[0]]) + 1 for comp in comparisons]
                    
                    print(cat_fltr, targets_[cat_fltr], comparisons_[cat_fltr])
                
                # compute relative light curve for current filter
                relative_light_curves[cat_fltr], transformed_masks[cat_fltr] = self._compute_relative_light_curve(cat_fltr, targets_[cat_fltr], comparisons_[cat_fltr], phot_type, self.t_ref, show_diagnostics)
            
            # plot the relative light curves for each filter
            self._plot_relative_light_curves(relative_light_curves, self.t_ref, transformed_masks, targets_, comparisons_, prefix, save_label)
            
            if self.show_plots:
                plt.show()
            
            # save relative light curves
            for (k, v) in relative_light_curves.items():
                v.to_csv(self.out_directory + "relative_light_curves/" + f"{prefix}_{k}_{save_label}.csv", index=False)
            
            return Analyser(relative_light_curves, self.out_directory, prefix, phot_type)
    
    def _compute_relative_light_curve(self, fltr: str, target: int, comparisons: List[int],
                                      phot_type: Literal["aperture", "annulus", "normal", "optimal"],
                                      t_ref: float, show_diagnostics: bool) -> Tuple[Dict[str, ArrayLike], ArrayLike]:
        """
        Compute the relative light curve for a target source with respect to one or more comparison sources for a given
        filter.

        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve for.
        target : int
            The catalog ID of the target source.
        comparisons : List[int]
            The catalog ID(s) of the comparison source(s).
        phot_type : Literal['aperture', 'annulus';, 'normal', 'optimal']
            The type of photometry to use.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        show_diagnostics : bool
            Whether to show diagnostic plots for each comparison source.

        Returns
        -------
        Tuple[Dict[str, ArrayLike], ArrayLike]
            The relative light curve and the transformation mask.
        """
        
        if phot_type == "aperture":
            save_label = "aperture_light_curve"
            light_curve_dir = "aperture_light_curves"
        elif phot_type == "annulus":
            save_label = "annulus_light_curve"
            light_curve_dir = "annulus_light_curves"
        elif phot_type == "normal":
            save_label = "normal_light_curve"
            light_curve_dir = "normal_light_curves"
        elif phot_type == "optimal":
            save_label = "optimal_light_curve"
            light_curve_dir = "optimal_light_curves"
        else:
            print(f"[OPTICAM] Flux type {phot_type} is not supported.")
            return None
        
        # define relative light curve dictionary
        relative_light_curve = {
            'MJD': [],
            'BDT': [],
            'relative flux': [],
            'relative flux error': [],
        }
        if phot_type == 'aperture' or phot_type == 'annulus':
            relative_light_curve.update({'quality_flag': []})
        
        # get target data frame
        try:
            target_df = pd.read_csv(self.out_directory + light_curve_dir + '/' + fltr + '_source_' + str(target) + '.csv')
        except:
            print('[OPTICAM] Could not load ' + light_curve_dir + '/' + fltr + '_source_' + str(target) + '.csv, skipping ...')
            return None
        
        # get comparison data frames
        comp_dfs = []
        for comp in comparisons:
            try:
                comparison_df = pd.read_csv(self.out_directory + light_curve_dir + '/' + fltr + '_source_' + str(comp) + '.csv')
            except:
                print('[OPTICAM] Could not load ' + light_curve_dir + '/' + fltr + '_source_' + str(comp) + '.csv, skipping ...')
                continue
            comp_dfs.append(comparison_df)
        
        # get time columns from all data frames
        time_columns = [target_df["MJD"].values]
        time_columns.extend([df["MJD"].values for df in comp_dfs])
        
        # get matching times between all data frames
        common_times = set(time_columns[0])
        for time_col in time_columns[1:]:
            common_times.intersection_update(time_col)
        common_times = sorted(common_times)
        
        # get matching times for target
        filtered_target_df = target_df[target_df["MJD"].isin(common_times)]
        filtered_target_df.reset_index(drop=True, inplace=True)
        
        # get matching times for comparisons
        filtered_comp_dfs = [df[df["MJD"].isin(common_times)] for df in comp_dfs]
        filtered_comp_dfs = [df.reset_index(drop=True) for df in filtered_comp_dfs]
        
        # define transform mask
        if "quality_flag" in filtered_target_df.columns:
            transformed_mask = filtered_target_df["quality_flag"] == "A"
        else:
            # if no quality flag column is present flux are from normal or optimal photometry, which are all aligned
            transformed_mask = np.ones_like(filtered_target_df["flux"].values, dtype=bool)
        
        # plot diagnostic light curves
        for i, df in enumerate(filtered_comp_dfs):
            self._plot_raw_diag(fltr, target, comparisons[i], filtered_target_df, df, t_ref, save_label, show_diagnostics)
        for i, df in enumerate(filtered_comp_dfs):
            for j, df2 in enumerate(filtered_comp_dfs):
                if i != j:
                    self._plot_diag(fltr, comparisons[i], comparisons[j], df, df2, t_ref, save_label, show_diagnostics)
        
        # get total flux and error of comparison sources
        comp_fluxes = np.sum([df["flux"].values for df in filtered_comp_dfs], axis=0)
        comp_flux_errors = np.sqrt(np.sum([np.square(df["flux_error"].values) for df in filtered_comp_dfs], axis=0))
        
        relative_light_curve['relative flux'] = filtered_target_df["flux"].values/comp_fluxes
        relative_light_curve['relative flux error'] = relative_light_curve['relative flux']*np.abs(np.sqrt(np.square(filtered_target_df["flux_error"].values/filtered_target_df["flux"].values) + np.square(comp_flux_errors/comp_fluxes)))
        
        # add quality flag column if it exists
        try:
            relative_light_curve['quality_flag'] = filtered_target_df["quality_flag"].values
        except:
            pass
        
        relative_light_curve['MJD'] = filtered_target_df["MJD"].values
        relative_light_curve['BDT'] = filtered_target_df["BDT"].values
        
        return pd.DataFrame(relative_light_curve), transformed_mask
    
    def _plot_relative_light_curve(self, relative_light_curve: Dict[str, ArrayLike], t_ref: float,
                                   transformed_mask: ArrayLike, ax = None, target: int = None,
                                   comparisons: List[int] = None, prefix: str = None, fltr: str = None,
                                   save_label: str = None) -> None:
        """
        Plot the relative light curve for a target source with respect to one or more comparison sources for a given filter.

        Parameters
        ----------
        relative_light_curve : Dict[str, ArrayLike]
            The relative light curve.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        transformed_mask : ArrayLike
            The transformation mask. Unaligned data points are plotted in red.
        ax : _type_, optional
            The axis onto which the relative light curve is plotted, by default None (a new figure is created).
        target : int, optional
            The catalog ID of the target source, by default None.
        comparisons : List[int], optional
            The catalog ID(s) of the comparison source(s), by default None.
        prefix : str, optional
            The prefix to use when saving the relative light curve (e.g., the target star's name), by default None.
        fltr : str, optional
            The filter used 
        save_label : str, optional
            _description_, by default None
        """
        
        time = relative_light_curve["MJD"].copy()
        time -= t_ref
        time *= 86400
        
        # if an axis is not provided, create a new figure
        # an axis will be provided if light curves for multiple filters are being plotted
        # if only plotting a single filter, a dedicated figure is created
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=(1.5*6.4, 4.8))
            dedicated_plot = True
        else:
            dedicated_plot = False
        
        ax.errorbar(time[transformed_mask], relative_light_curve['relative flux'][transformed_mask], np.abs(relative_light_curve['relative flux error'][transformed_mask]), fmt="k.", ms=2, ecolor="grey", elinewidth=1)
        ax.errorbar(time[~transformed_mask], relative_light_curve['relative flux'][~transformed_mask], np.abs(relative_light_curve['relative flux error'][~transformed_mask]), fmt="r.", ms=2, elinewidth=1, alpha=.2)
        
        ax.set_title(fltr + ' (Source ID: ' + str(target) + ', Comparison ID(s): ' + ', '.join([str(comp) for comp in comparisons]) + ')')
        
        if dedicated_plot:
            ax.set_xlabel(f"Time from MJD {t_ref:.4f} [s]")
            ax.set_ylabel("Relative flux")
        
            fig.savefig(self.out_directory + "relative_light_curves/" + f"{prefix}_{fltr}_{save_label}.png")
        
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)
    
    def _plot_relative_light_curves(self, relative_light_curves: Dict[str, Dict[str, ArrayLike]], t_ref: float,
                                    transformed_masks: Dict[str, ArrayLike], targets: Dict[str, int],
                                    comparisons: Dict[str, List[int]], prefix: str, save_label: str) -> None:
        """
        Plot the relative light curves for a target source with respect to one or more comparison sources for multiple filters.

        Parameters
        ----------
        relative_light_curves : Dict[str, Dict[str, ArrayLike]]
            The relative light curves for each filter.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        transformed_masks : Dict[str, ArrayLike]
            The transformation masks for each filter.
        targets : Dict[str, int]
            The catalog ID of the target source for each filter.
        comparisons : Dict[str, List[int]]
            The catalog ID(s) of the comparison source(s) for each filter.
        prefix : str
            The prefix to use when saving the relative light curve (e.g., the target star's name).
        save_label : str
            The label to use when saving the relative light curve.
        """
        
        fig, axs = plt.subplots(nrows=len(relative_light_curves), tight_layout=True, sharex=True,
                                figsize=(1.5 * 6.4, 2 * len(relative_light_curves) / 3 * 4.8))
        
        for fltr, relative_light_curve in relative_light_curves.items():
            self._plot_relative_light_curve(relative_light_curve, t_ref, transformed_masks[fltr], axs[self.filters.index(fltr)], targets[fltr], comparisons[fltr], prefix, fltr, save_label)
        
        axs[-1].set_xlabel(f"Time from MJD {t_ref:.4f} [s]")
        axs[int(len(relative_light_curves) / 2)].set_ylabel("Relative flux")
        
        fig.savefig(self.out_directory + "relative_light_curves/" + f"{prefix}_{save_label}.png")
        
        if self.show_plots:
            plt.show(fig)
        else:
            plt.close(fig)
    
    def _plot_raw_diag(self, fltr: str, target: int, comparison: int, target_df: pd.DataFrame, comparison_df: pd.DataFrame,
                   t_ref: float, save_label: str, show: bool) -> None:
        """
        Plot the raw light curves for a target and comparison source for a given filter.
        
        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve for.
        target : int
            The catalog ID of the target source.
        comparison : int
            The catalog ID of the comparison source.
        target_df : pd.DataFrame
            The data frame of the target source.
        comparison_df : pd.DataFrame
            The data frame of the comparison source.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        save_label : str
            The label to use when saving the diagnostic plot.
        show : bool
            Whether to show the diagnostic plot.
        """
        
        time = (target_df["MJD"].values - t_ref)*86400  # convert to seconds
        
        diag_fig, diag_ax = plt.subplots(nrows=2, tight_layout=True, sharex=True, figsize=(6.4, 1.5*4.8), gridspec_kw={"height_ratios": [2, 1], "hspace": 0})
        
        diag_ax[0].set_title(fltr + ' Source ID: ' + str(target) + ', Comparison ID: ' + str(comparison))
        
        diag_ax[0].plot(time, target_df["flux"]/target_df["flux"].median(), "k-", lw=1, label="Target")
        diag_ax[0].plot(time, comparison_df["flux"]/comparison_df["flux"].median(), "r-", alpha=.5, lw=1, label="Comparison")
        
        diag_ax[1].plot(time, target_df["flux"]/target_df["flux"].median() - comparison_df["flux"]/comparison_df["flux"].median(), "k-", lw=1)
        
        diag_ax[0].set_ylabel("Normalised raw flux [counts]")
        diag_ax[0].xaxis.set_tick_params(labelbottom=False)
        diag_ax[0].legend()
        diag_ax[1].set_ylabel("Residuals")
        diag_ax[1].set_xlabel(f"Time from MJD {t_ref} [s]")
        
        for diag_ax in diag_ax:
            diag_ax.minorticks_on()
            diag_ax.tick_params(which="both", direction="in", top=True, right=True)
        
        if not os.path.isdir(self.out_directory + "relative_light_curves/diag"):
            os.makedirs(self.out_directory + "relative_light_curves/diag", exist_ok=True)
        
        diag_fig.savefig(self.out_directory + 'relative_light_curves/diag/' + fltr + '_' + str(target) + '_' + str(comparison) + '_' + save_label + '_diag.png')
        
        if not show:
            diag_fig.clear()
            plt.close(diag_fig)
    
    def _plot_diag(self, fltr: str, comparison1: int, comparison2: int, comparison1_df: pd.DataFrame,
                   comparison2_df: pd.DataFrame, t_ref: float, save_label: str, show: bool) -> None:
        """
        Plot the relative diagnostic light curve for two comparison sources for a given filter.
        
        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve for.
        comparison1 : int
            The catalog ID of the first comparison source.
        comparison2 : int
            The catalog ID of the second comparison source.
        comparison1_df : pd.DataFrame
            The data frame of the first comparison source.
        comparison2_df : pd.DataFrame
            The data frame of the second comparison source.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        save_label : str
            The label to use when saving the diagnostic plot.
        show : bool
            Whether to show the diagnostic plot.
        """
        
        time = (comparison1_df["MJD"].values - t_ref)*86400  # convert to seconds
        
        diag_fig, diag_ax = plt.subplots(tight_layout=True, figsize=(6.4, 4.8))
        
        relative_flux = comparison1_df["flux"]/comparison2_df["flux"]
        relative_flux_error = relative_flux*np.abs(np.sqrt(np.square(comparison1_df["flux_error"].values/comparison1_df["flux"].values) + np.square(comparison2_df["flux_error"].values/comparison2_df["flux"].values)))
        
        # save light curve to CSV
        diag_df = pd.DataFrame({"MJD": comparison1_df["MJD"].values, "BDT":  comparison1_df["BDT"].values,
                                "relative flux": relative_flux, "relative flux error": relative_flux_error})
        diag_df.to_csv(self.out_directory + 'relative_light_curves/diag/' + fltr + '_' + str(comparison1) + '_' + str(comparison2) + '_' + save_label + '_diag.csv', index=False)
        
        diag_ax.errorbar(time, relative_flux/relative_flux.median(), np.abs(relative_flux_error/relative_flux.median()),
                         fmt="k.", ms=2, ecolor="grey", elinewidth=1)
        
        diag_ax.axhline(1, color="r", ls="-", lw=1, zorder=3)
        
        diag_ax.set_title(fltr + ' Comparison ID: ' + str(comparison1) + ', Comparison ID: ' + str(comparison2))
        diag_ax.set_xlabel(f"Time from MJD {t_ref} [s]")
        diag_ax.set_ylabel("Normalised relative flux")
        
        diag_ax.minorticks_on()
        diag_ax.tick_params(which="both", direction="in", top=True, right=True)
        
        if not os.path.isdir(self.out_directory + "relative_light_curves/diag"):
            os.makedirs(self.out_directory + "relative_light_curves/diag", exist_ok=True)
        
        diag_fig.savefig(self.out_directory + 'relative_light_curves/diag/' + fltr + '_' + str(comparison1) + '_' + str(comparison2) + '_' + save_label + '_diag.png')
        
        if not show:
            diag_fig.clear()
            plt.close(diag_fig)