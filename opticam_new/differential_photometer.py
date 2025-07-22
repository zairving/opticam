from typing import Dict, List, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import json
from stingray import Lightcurve
import matplotlib.colors as mcolors
from matplotlib.axes import Axes

from opticam_new.analyser import Analyser
from opticam_new.helpers import plot_catalogue, infer_gtis

# TODO: it should be possible to improve camera matching by taking into account the pixel-scale and FoV differences between the cameras.

class DifferentialPhotometer:
    """
    Helper class for creating relative light curves from OPTICAM data.
    """
    
    def __init__(
        self,
        out_directory: str,
        show_plots: bool = True,
        ) -> None:
        """
        Helper class for creating relative light curves from OPTICAM data.
        
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
        
        ########################################### input params ###########################################
        
        self.out_directory = out_directory
        if not os.path.isdir(self.out_directory):
            raise FileNotFoundError(f'[OPTICAM] {self.out_directory} not found.')
        
        self.show_plots = show_plots
        
        ########################################### attributes ###########################################
        
        with open(os.path.join(self.out_directory, 'misc/input_parameters.json'), 'r') as file:
            input_parameters = json.load(file)
        self.filters = input_parameters['filters']
        self.t_ref = input_parameters['t_ref']
        
        # output filters
        print('[OPTICAM] Filters: ' + ', '.join(list(self.filters)))
        
        ########################################### read catalogues ###########################################
        
        self.catalogues = {}
        for fltr in self.filters:
            try:
                self.catalogues.update(
                    {
                        fltr: QTable.read(
                            os.path.join(self.out_directory, f'cat/{fltr}_catalogue.ecsv'),
                            format='ascii.ecsv'),
                        },
                    )
            except:
                print(f'[OPTICAM] Could not load {os.path.join(self.out_directory, f'cat/{fltr}_catalogue.ecsv')}, skipping ...')
                self.filters.remove(fltr)
                continue
        
        ########################################### plot catalogues ###########################################
        
        if show_plots:
            stacked_images = {}
            for fltr in self.filters:
                path = os.path.join(self.out_directory, f'cat/{fltr}_stacked_image.npz')
                try:
                    stacked_images[fltr] = np.load(path)['stacked_image']
                except:
                    print(f"[OPTICAM] Could not load {path}, skipping ...")
            
            colours = list(mcolors.TABLEAU_COLORS.keys())
            colours.pop(colours.index("tab:brown"))
            colours.pop(colours.index("tab:gray"))
            colours.pop(colours.index("tab:purple"))
            colours.pop(colours.index("tab:blue"))
            
            fig, ax = plot_catalogue(
                self.filters,
                stacked_images,
                self.catalogues,
                colours,
                )
            
            plt.show()
    
    def get_relative_light_curve(
        self,
        fltr: str,
        target: int,
        comparisons: List[int],
        phot_label: str,
        prefix: str | None = None,
        match_other_cameras = False,
        show_diagnostics: bool = True,
        ) -> Analyser:
        """
        Compute the relative light curve for a target source with respect to one or more comparison sources. By default,
        the relative light curve is computed for a single filter. The relative light curve is saved to
        out_directory/relative_light_curves. To automatically match the target and comparison sources across the other
        two filters, set match_other_cameras to True. Note that this can incorrectly match sources, so it is recommended
        to manually check the results.
        
        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve for.
        target : int
            The catalogue ID of the target source.
        comparison : List[int]
            The catalogue ID(s) of the comparison source(s). Even if a single comparison source is used, this should
            be a list of length 1.
        phot_label : str
            The photometry label, used for file reading and labelling.
        prefix : str, optional
            The prefix to use when saving the relative light curve (e.g., the target star's name), by default None.
        match_other_cameras : bool, optional
            Whether to try and automatically match the target and comparison sources across OPTICAM's other cameras, by
            default False. Note that this can incorrectly match sources, particularly if the fields are crowded, and so
            it is recommended to manually check the results.
        show_diagnostics : bool, optional
            Whether to show diagnostic plots, by default True.
        
        Returns
        -------
        Analyser
            An Analyser object containing the relative light curve(s).
        """
        
        save_label = f'{phot_label}_light_curve'
        
        if not os.path.isdir(os.path.join(self.out_directory, 'relative_light_curves')):
            os.mkdir(os.path.join(self.out_directory, 'relative_light_curves'))
        
        # validate filter
        if fltr not in self.filters:
            raise ValueError('[OPTICAM] ' + fltr + ' is not a valid filter.')
        
        if not match_other_cameras:
            # compute and plot relative light curve for single filter
            relative_light_curve = self._compute_relative_light_curve(
                fltr,
                target,
                comparisons,
                phot_label,
                self.t_ref,
                show_diagnostics,
                )
            
            if relative_light_curve is None:
                raise ValueError(f"[OPTICAM] Could not compute relative light curve for filter {fltr}, target {target}, comparisons {comparisons}.")
            
            self._plot_relative_light_curve(
                relative_light_curve,
                self.t_ref,
                target=target,
                comparisons=comparisons,
                prefix=prefix,
                fltr=fltr,
                save_label=save_label,
                )
            
            # save relative light curve to CSV
            df = pd.DataFrame({
                "TDB": relative_light_curve.time,
                "relative flux": relative_light_curve.counts,
                "relative flux error": relative_light_curve.counts_err
            })
            df.to_csv(
                os.path.join(self.out_directory, f'relative_light_curves/{prefix}_{fltr}_{save_label}.csv'),
                index=False,
            )
            
            # return Analyser object
            return Analyser({fltr: relative_light_curve}, self.out_directory, prefix, phot_label)
        else:
            # define dictionaries to store relative light curves for each camera
            relative_light_curves = {}
            targets_ = {}
            comparisons_ = {}
            
            # get source coordinates for input filter
            input_filter_coords = np.array([self.catalogues[fltr]['xcentroid'], self.catalogues[fltr]['ycentroid']]).T
            
            for cat_fltr in self.filters:
                # get target and comparison source indices
                if cat_fltr == fltr:
                    # if the current filter is the input filter, the target and comparison sources are already known
                    targets_[cat_fltr] = target
                    comparisons_[cat_fltr] = comparisons
                else:
                    # if the current filter is not the input filter, the target and comparison sources need to be matched
                    # using the Hungarian algorithm
                    fltr_coords = np.array([self.catalogues[cat_fltr]['xcentroid'], self.catalogues[cat_fltr]['ycentroid']]).T  # get source coordinates for current filter
                    distance_matrix = cdist(input_filter_coords, fltr_coords)  # compute distance matrix
                    input_filter_indices, fltr_indices = linear_sum_assignment(distance_matrix)  # solve assignment problem
                    
                    # get target and comparison source indices
                    targets_[cat_fltr] = int(fltr_indices[np.where(input_filter_indices == target - 1)[0]]) + 1
                    comparisons_[cat_fltr] = [int(fltr_indices[np.where(input_filter_indices == comp - 1)[0]]) + 1 for comp in comparisons]
                    
                    print(f'[OPTICAM] Matched {fltr} source {target} to {cat_fltr} source {targets_[cat_fltr]}.')
                    for i in range(len(comparisons)):
                        print(f'[OPTICAM] Matched {fltr} source {comparisons[i]} to {cat_fltr} source {comparisons_[cat_fltr][i]}')
                
                # compute relative light curve for current filter
                relative_light_curves[cat_fltr] = self._compute_relative_light_curve(
                    cat_fltr,
                    targets_[cat_fltr],
                    comparisons_[cat_fltr],
                    phot_label,
                    self.t_ref, 
                    show_diagnostics,
                    )
            
            # plot the relative light curves for each filter
            self._plot_relative_light_curves(
                relative_light_curves,
                self.t_ref,
                targets_,
                comparisons_,
                prefix,
                save_label,
                )
            
            # save relative light curves
            for (k, v) in relative_light_curves.items():
                df = pd.DataFrame({
                    "TDB": v.time,
                    "relative flux": v.counts,
                    "relative flux error": v.counts_err
                })
                df.to_csv(os.path.join(self.out_directory, f'relative_light_curves/{prefix}_{k}_{save_label}.csv'),
                                       index=False)
            
            return Analyser(relative_light_curves, self.out_directory, prefix, phot_label)
    
    def _compute_relative_light_curve(
        self,
        fltr: str,
        target: int,
        comparisons: List[int],
        phot_label: str,
        t_ref: float,
        show_diagnostics: bool,
        ) -> Lightcurve | None:
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
        Lightcurve
            The relative light curve (as a Stingray Lightcurve object).
        """
        
        # subdirectory where results will be saved
        light_curve_dir = f'{phot_label}_light_curves'
        
        # get target data frame
        try:
            target_df = pd.read_csv(os.path.join(self.out_directory, f'{light_curve_dir}/{fltr}_source_{target}.csv'))
        except:
            print(
                f'[OPTICAM] Could not load {os.path.join(self.out_directory,
                f'{light_curve_dir}/{fltr}_source_{target}.csv')}, skipping ...',
                )
            return None
        
        # get comparison data frame(s)
        comp_dfs = []
        for comp in comparisons:
            path = os.path.join(self.out_directory, f'{light_curve_dir}/{fltr}_source_{comp}.csv')
            try:
                comparison_df = pd.read_csv(path)
            except:
                print(f'[OPTICAM] Could not load {path}, skipping ...')
                continue
            comp_dfs.append(comparison_df)
        
        # ensure all light curve DataFrames have a common time column
        filtered_target_df, filtered_comp_dfs = self._filter_dataframes_to_common_time_column(
            target_df,
            comp_dfs,
            )
        
        # plot diagnostic light curves for target and comparison source(s)
        for i, df in enumerate(filtered_comp_dfs):
            self._plot_diag(
                fltr,
                target,
                comparisons[i],
                filtered_target_df,
                df,
                phot_label,
                t_ref,
                show_diagnostics,
            )
        
        if len(filtered_comp_dfs) > 1:
            # plot diagnostic light curves between comparison sources
            for i, df in enumerate(filtered_comp_dfs):
                for j, df2 in enumerate(filtered_comp_dfs):
                    if i != j:
                        self._plot_diag(
                            fltr,
                            comparisons[i],
                            comparisons[j],
                            df,
                            df2,
                            phot_label,
                            t_ref,
                            show_diagnostics,
                            )
        
        # get total flux and error of comparison sources
        comp_fluxes = np.sum([df["flux"].values for df in filtered_comp_dfs], axis=0)
        comp_flux_errors = np.sqrt(np.sum([np.square(df["flux_error"].values) for df in filtered_comp_dfs], axis=0))
        
        time = filtered_target_df["TDB"].values
        
        # compute relative flux and error
        relative_flux = filtered_target_df["flux"].values / comp_fluxes
        relative_flux_error = relative_flux * np.abs(np.sqrt(np.square(filtered_target_df["flux_error"].values / filtered_target_df["flux"].values) + np.square(comp_flux_errors / comp_fluxes)))
        
        # mask non-finite values
        valid_mask = np.isfinite(relative_flux) & np.isfinite(relative_flux_error)
        time = time[valid_mask]
        relative_flux = relative_flux[valid_mask]
        relative_flux_error = relative_flux_error[valid_mask]
        
        # infer light curve GTIs
        gtis = infer_gtis(time, threshold=1.5)
        
        return Lightcurve(time, relative_flux, err=relative_flux_error, gti=gtis)
    
    def _filter_dataframes_to_common_time_column(
        self,
        target_df: pd.DataFrame,
        comp_dfs: List[pd.DataFrame],
        ) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
        """
        Get the matching times between a target data frame (light curve) and a list of comparison data frames (light 
        curves).
        
        Parameters
        ----------
        target_df : pd.DataFrame
            The data frame of the target source.
        comp_dfs : List[pd.DataFrame]
            The list of data frames of the comparison sources.
        
        Returns
        -------
        Tuple[pd.DataFrame, List[pd.DataFrame]]
            The filtered target data frame and the list of filtered comparison data frames.
        """
        
        # get time columns from all data frames
        time_columns = [target_df['TDB'].values]
        time_columns.extend([df['TDB'].values for df in comp_dfs])
        
        # get matching times between all data frames
        common_times = set(time_columns[0])
        for time_col in time_columns[1:]:
            common_times.intersection_update(time_col)
        common_times = sorted(common_times)
        
        # get matching times for target
        filtered_target_df = target_df[target_df['TDB'].isin(common_times)]
        filtered_target_df.reset_index(drop=True, inplace=True)
        
        # get matching times for comparisons
        filtered_comp_dfs = [df[df['TDB'].isin(common_times)] for df in comp_dfs]
        filtered_comp_dfs = [df.reset_index(drop=True) for df in filtered_comp_dfs]
        
        return filtered_target_df, filtered_comp_dfs
    
    def _plot_relative_light_curve(
        self,
        relative_light_curve: Lightcurve,
        t_ref: float,
        target: int,
        comparisons: List[int],
        prefix: str | None,
        fltr: str,
        save_label: str,
        ax: Axes | None = None,
        ) -> None:
        """
        Plot the relative light curve for a target source with respect to one or more comparison sources for a given
        filter.
        
        Parameters
        ----------
        relative_light_curve : Lightcurve
            The relative light curve.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        target : int
            The catalog ID of the target source.
        comparisons : List[int]
            The catalog ID(s) of the comparison source(s).
        prefix : str | None
            The prefix to use when saving the relative light curve (e.g., the target star's name).
        fltr : str
            The filter.
        save_label : str
            The label to use when saving the relative light curve.
        ax : Axes, optional
            The axes to plot the relative light curve on. If None, a new figure and axes will be created.
        """
        
        # convert time to seconds from t_ref
        time = np.asarray(relative_light_curve.time).copy()
        time -= t_ref
        time *= 86400
        
        if ax is None:
            fig, ax = plt.subplots(tight_layout=True, figsize=(1.5 * 6.4, 4.8))
            standalone_plot = True
        else:
            standalone_plot = False
        
        ax.errorbar(
            time,
            relative_light_curve.counts,
            np.abs(relative_light_curve.counts_err),
            fmt="k.",
            ms=2,
            ecolor="grey",
            elinewidth=1,
            )
        
        if standalone_plot:
            ax.set_title(f'{fltr} (Source ID: {target}, Comparison ID(s): {', '.join([str(comp) for comp in comparisons])})')
            ax.set_xlabel(f"Time from TDB {t_ref:.4f} [s]")
            ax.set_ylabel("Relative flux")
            
            fig.savefig(os.path.join(self.out_directory, f'relative_light_curves/{prefix}_{fltr}_{save_label}.png'))
            
            if self.show_plots:
                plt.show()
            else:
                fig.clear()
                plt.close(fig)
    
    def _plot_relative_light_curves(
        self,
        relative_light_curves: Dict[str, Lightcurve],
        t_ref: float,
        targets: Dict[str, int],
        comparisons: Dict[str, List[int]],
        prefix: str | None,
        save_label: str,
        ) -> None:
        """
        Plot the relative light curves for a target source with respect to one or more comparison sources for multiple filters.

        Parameters
        ----------
        relative_light_curves : Dict[str, Dict[str, ArrayLike]]
            The relative light curves for each filter.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        targets : Dict[str, int]
            The catalog ID of the target source for each filter.
        comparisons : Dict[str, List[int]]
            The catalog ID(s) of the comparison source(s) for each filter.
        prefix : str | None
            The prefix to use when saving the relative light curve (e.g., the target star's name).
        save_label : str
            The label to use when saving the relative light curve.
        """
        
        fig, axes = plt.subplots(nrows=len(relative_light_curves), tight_layout=True, sharex=True,
                                figsize=(1.5 * 6.4, 2 * len(relative_light_curves) / 3 * 4.8))
        
        for fltr, relative_light_curve in relative_light_curves.items():
            self._plot_relative_light_curve(
                relative_light_curve,
                t_ref,
                targets[fltr],
                comparisons[fltr],
                prefix,
                fltr,
                save_label,
                axes[self.filters.index(fltr)],
                )
        
        axes[-1].set_xlabel(f"Time from TDB {t_ref:.4f} [s]")
        axes[int(len(relative_light_curves) / 2)].set_ylabel("Relative flux")
        
        fig.savefig(os.path.join(self.out_directory, f'relative_light_curves/{prefix}_{save_label}.png'))
        
        if self.show_plots:
            plt.show(fig)
        else:
            fig.clear()
            plt.close(fig)
    
    def _plot_diag(
        self,
        fltr: str,
        comparison1: int,
        comparison2: int,
        comparison1_df: pd.DataFrame,
        comparison2_df: pd.DataFrame,
        phot_label: str,
        t_ref: float,
        show: bool,
        ) -> None:
        """
        Plot the relative diagnostic light curve for two comparison sources for a given filter.
        
        Parameters
        ----------
        fltr : str
            The filter to compute the relative light curve.
        comparison1 : int
            The catalog ID of the first comparison source.
        comparison2 : int
            The catalog ID of the second comparison source.
        comparison1_df : pd.DataFrame
            The data frame of the first comparison source.
        comparison2_df : pd.DataFrame
            The data frame of the second comparison source.
        pho_label : str
            The photometry label.
        t_ref : float
            The time of the earliest observation (used for plotting the relative light curve in seconds from t_ref).
        show : bool
            Whether to show the diagnostic plot.
        """
        
        # convert time to seconds from t_ref
        time = (comparison1_df["TDB"].values - t_ref) * 86400
        
        fig, axes = plt.subplots(
            nrows=3,
            tight_layout=True,
            sharex=True,
            figsize=(6.4, 2 * 4.8),
            gridspec_kw={
                "hspace": 0,
                },
            )
        
        # compute relative flux and error
        relative_flux = comparison1_df["flux"].values / comparison2_df["flux"].values
        relative_flux_error = relative_flux * np.abs(np.sqrt(np.square(comparison1_df["flux_error"].values / comparison1_df["flux"].values) + np.square(comparison2_df["flux_error"].values / comparison2_df["flux"].values)))
        
        ########################################### normalised light curves ###########################################
        
        axes[0].set_title(f'{fltr} Comparison ID: {comparison1}, Comparison ID: {comparison2}')
        axes[0].errorbar(
            time,
            comparison1_df["flux"] / comparison1_df["flux"].median(),
            np.abs(comparison1_df["flux_error"] / comparison1_df["flux"].median()),
            fmt="kx-",
            ms=5,
            elinewidth=1,
            label=f'Source {comparison1}',
            alpha=.5,
            )
        axes[0].errorbar(
            time,
            comparison2_df["flux"] / comparison2_df["flux"].median(),
            np.abs(comparison2_df["flux_error"] / comparison2_df["flux"].median()),
            fmt="r+-",
            ms=5,
            elinewidth=1,
            label=f'Source {comparison2}',
            alpha=.5,
            )
        axes[0].legend()
        axes[0].set_ylabel("Normalised raw flux [counts]")
        
        ########################################### residuals ###########################################
        
        axes[1].errorbar(
            time,
            comparison1_df["flux"] / comparison1_df["flux"].median() - comparison2_df["flux"] / comparison2_df["flux"].median(),
            np.sqrt(np.sum(
                [
                    np.square(comparison1_df["flux_error"].values / comparison1_df["flux"].values),
                    np.square(comparison2_df["flux_error"].values / comparison2_df["flux"].values),
                ],
                axis=0,
                ),
                    ),
            fmt='k.',
            ms=2,
            ecolor='grey',
            elinewidth=1,
            )
        axes[1].axhline(
            0,
            color='r',
            lw=1,
        )
        axes[1].set_ylabel('Residuals')
        
        ########################################### relative light curve ###########################################
        
        axes[2].errorbar(
            time,
            relative_flux / np.median(relative_flux),
            np.abs(relative_flux_error) / np.median(relative_flux),
            fmt="k.",
            ms=2,
            ecolor="grey",
            elinewidth=1,
            )
        axes[2].axhline(
            1,
            color='r',
            lw=1,
        )
        axes[2].set_xlabel(f"Time from TDB {t_ref} [s]")
        axes[2].set_ylabel("Normalised relative flux")
        
        ########################################### format plot ###########################################
        
        for ax in axes:
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
        ########################################### save plot ###########################################
        
        if not os.path.isdir(os.path.join(self.out_directory, 'relative_light_curves/diag')):
            os.makedirs(os.path.join(self.out_directory, 'relative_light_curves/diag'))
        
        fig.savefig(os.path.join(self.out_directory, f'relative_light_curves/diag/{fltr}_{comparison1}_{comparison2}_{phot_label}.png'))
        
        ########################################### optionally show plot ###########################################
        
        if not show:
            fig.clear()
            plt.close(fig)