import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from typing import Dict, Literal, Tuple
from numpy.typing import NDArray
import os
import astropy.units as u
from astropy.units.quantity import Quantity
import json
import warnings


class Analyser:
    """
    Helper class for analysing OPTICAM light curves.
    """
    
    def __init__(self, light_curves: Dict[str, pd.DataFrame], out_directory: str, prefix: str, phot_type: str):
        """
        Helper class for analysing OPTICAM light curves.
        
        Parameters
        ----------
        light_curves : Dict[str, pd.DataFrame]
            The light curves to analyse, where the keys are the filter names and the values are the light curves.
        out_directory : str
            The directory to save the output files. This should be the same as the `out_dir` used during data reduction
            (i.e., opticam_new.Reducer()` since `Analyser` assumes a certain directory structure and requires certain
            files created by `Reducer()`.
        prefix : str
            The prefix to use for the output files (e.g., the name of the target).
        phot_type : str
            The type of photometry used to generate the light curves. This is only used for naming the output files.
        """
        
        self.light_curves = light_curves
        
        # drop NaNs to avoid issues with methods
        for lc in light_curves.values():
            lc.dropna(inplace=True)
            lc.reset_index(drop=True, inplace=True)  # reset index after dropping NaNs
        
        self.filters = list(light_curves.keys())
        self.out_directory = out_directory
        self.prefix = prefix
        self.phot_type = phot_type
        
        with open(os.path.join(self.out_directory, 'misc/input_parameters.json'), 'r') as file:
            input_parameters = json.load(file)
        self.t_ref_mjd = input_parameters['t_ref_mjd']
        self.t_ref_bdt = input_parameters['t_ref_bdt']
        
        self.colours = {
            'g-band': 'tab:green',  # camera 1
            'u-band': 'tab:green',  # camera 1
            'r-band': 'tab:orange',  # camera 2
            'i-band': 'tab:olive',  # camera 3
            'z-band': 'tab:olive',  # camera 3
        }
    
    def plot(self, title: str | None = None, time: Literal['MJD', 'BDT'] = 'MJD', ax = None) -> plt.Figure:
        """
        Plot the light curves.
        
        Parameters
        ----------
        title : str, optional
            _description_, by default None
        time : Literal['MJD', 'BDT'], optional
            Time axis, by default 'MJD'.
        ax : _type_, optional
            _description_, by default None
        
        Returns
        -------
        plt.Figure
            The figure containing the light curves.
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(2 * 6.4, 4.8), tight_layout=True)
        
        if time == 'MJD':
            t_ref = self.t_ref_mjd
        elif time == 'BDT':
            t_ref = self.t_ref_bdt
        else:
            raise ValueError("[OPTICAM] when calling Analyser.plot(), time must be 'MJD' or 'BDT'")
        
        for fltr, lc in self.light_curves.items():
            
            t = (lc[time] - t_ref) * 86400
            
            ax.errorbar(t, lc['relative flux'], lc['relative flux error'], marker='.', ms=2,
                        linestyle='none', color=self.colours[fltr], ecolor='grey', elinewidth=1, label=fltr)
        
        ax.set_xlabel(f'Time from {time} {t_ref:.4f} [s]')
        ax.set_ylabel('Relative Flux')
        
        if title is not None:
            ax.set_title(title)
        
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
        
        ax.legend(markerscale=5)
        
        return fig
    
    def clip_outliers(self, n_window: int, sigma: float = 5., max_iters: int = 10) -> None:
        """
        Clip outliers using a rolling median filter.
        
        Parameters
        ----------
        n_window : int
            The size of the window (number of data points) to use for the rolling median filter.
        sigma : float, optional
            The clipping threshold in standard deviations, by default 5.
        max_iters : int, optional
            The maximum number of clipping iterations, by default 10. Fewer iterations may be performed if the clipping
            converges before reaching the maximum number of iterations.
        """
        
        for fltr, lc in self.light_curves.items():
            clipped_lc = lc.copy()
            
            for _ in range(max_iters):
                clipped_lc['median'] = clipped_lc['relative flux'].rolling(window=n_window, min_periods=1).median()
                clipped_lc['std_dev'] = clipped_lc['relative flux'].rolling(window=n_window, min_periods=1).std()
                
                clipped_lc['lower_bound'] = clipped_lc['median'] - sigma * clipped_lc['std_dev']
                clipped_lc['upper_bound'] = clipped_lc['median'] + sigma * clipped_lc['std_dev']
                
                clipped_lc = clipped_lc[(clipped_lc['relative flux'] >= clipped_lc['lower_bound']) & 
                                        (clipped_lc['relative flux'] <= clipped_lc['upper_bound'])]
                
                if len(clipped_lc) == len(lc):
                    break
            
            clipped_lc.reset_index(drop=True, inplace=True)
            self.light_curves[fltr] = clipped_lc.copy()
    
    def update(self, analyser: 'Analyser') -> None:
        """
        Combine another `Analyser` object with the current one. Useful when creating relative light curves
        individually per filter.
        
        Parameters
        ----------
        analyser : Analyser
            The analyser object being combined with the current one.
        """
        
        light_curves = analyser.light_curves
        filters = analyser.filters
        
        for fltr in filters:
            self.light_curves[fltr] = light_curves[fltr].copy()
    
    def lomb_scargle(self, frequencies: NDArray = None, scale: Literal['linear', 'log', 'loglog'] = 'linear', 
                     show_plot=True) -> Tuple[NDArray, Dict[str, NDArray]] | Dict[str, NDArray]:
        """
        Compute the Lomb-Scargle periodogram for each light curve.
        
        Parameters
        ----------
        frequencies : NDArray, optional
            The periodogram frequencies, by default None. If None, a suitable frequency range will be inferred from the
            data.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the inferred frequencies, by default 'linear'. If 'linear', the frequency grid is
            linearly spaced. If 'log', the frequency grid is logarithmically spaced. The upper and lower bounds of the
            frequencies are always the same. If 'loglog', both the frequency and power axes will be in logarithm.
        show_plot : bool, optional
            Whether to show the plot of the periodogram(s), by default True.
        Returns
        -------
        Tuple[NDArray, Dict[str, NDArray]] | Dict[str, NDArray]
            If no frequencies are provided, returns a tuple containing the frequencies and a dictionary of periodograms
            for each light curve. If frequencies are provided, returns a dictionary of periodograms for each light curve.
        """
        
        if not os.path.isdir(f'{self.out_directory}/periodograms'):
            os.mkdir(f'{self.out_directory}/periodograms')
        
        if frequencies is None:
            
            return_frequencies = True
            
            t_span = np.mean([lc['BDT'].max() - lc['BDT'].min() for lc in self.light_curves.values()]) * 86400
            dt = np.min([np.min(np.diff(lc['BDT'])) for lc in self.light_curves.values()]) * 86400
            
            lo = 1 / t_span
            hi = 0.5 / dt
            
            if scale == 'linear':
                frequencies = np.linspace(lo, hi, 10*round(hi/lo))
            elif scale == 'log' or scale == 'loglog':
                frequencies = np.logspace(np.log10(lo), np.log10(hi), 10*round(hi/lo))
        
        results = {}
        
        fig, axs = plt.subplots(tight_layout=True, nrows=len(self.light_curves), sharex=True,
                                figsize=(6.4, (0.5 + 0.5*len(self.light_curves)*4.8)), gridspec_kw={'hspace': 0.})
        
        for i in range(len(self.light_curves)):
            
            k = list(self.light_curves.keys())[i]
            power = LombScargle((self.light_curves[k]['BDT'].values - self.light_curves[k]['BDT'].min())*86400,
                                self.light_curves[k]['relative flux'],
                                self.light_curves[k]['relative flux error']).power(frequencies)
            results[k] = power
            
            axs[i].plot(frequencies, power, color=self.colours[k], lw=1, label=k)
            
            axs[i].text(.05, .1, k, transform=axs[i].transAxes, fontsize='large', ha='left')
        
        if scale == 'log':
            for ax in axs:
                ax.set_xscale('log')
        
        if scale == 'loglog':
            for ax in axs:
                ax.set_xscale('log')
                ax.set_yscale('log')
        
        def freq2period(x):
            return 1 / x
        
        def period2freq(x):
            return 1 / x
        
        for i, ax in enumerate(axs.flatten()):
            
            # ax.minorticks_on()
            ax.tick_params(which='both', direction='in', right=True)
            
            secax = ax.secondary_xaxis('top', functions=(freq2period, period2freq))
            # secax.minorticks_on()
            secax.tick_params(which='both', direction='in', top=True, labeltop=True if i == 0 else False)
            
            if i == 0:
                secax.set_xlabel('Period [s]')
        
        axs[-1].set_xlabel('Frequency [Hz]')
        axs[len(self.light_curves)//2].set_ylabel('Power')
        
        fig.savefig(f'{self.out_directory}/periodograms/{self.prefix}_{self.phot_type}_periodogram.png')
        
        if show_plot:
            plt.show()
        
        if return_frequencies:
            return frequencies, results
        
        return results
    
    def phase_fold(self, period: Quantity, plot=True) -> Dict[str, NDArray]:
        """
        Phase fold each light curve using the given period.
        
        Parameters
        ----------
        period : Quantity
            The period to use for phase folding. This must be an astropy `Quantity` with units of time (e.g.,
            `astropy.units.s`) to ensure correct handling of the period.
        plot : bool, optional
            Whether to plot the phase folded light curves, by default True.
        
        Returns
        -------
        Dict[str, NDArray]
            The phase folded light curves.
        """
        
        period = period.to(u.day).value  # convert from given units to days
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            phase = (lc['BDT'] % period) / period
            results[fltr] = phase
        
        if plot:
            fig, axs = plt.subplots(tight_layout=True, nrows=len(self.light_curves), sharex=True,
                                    figsize=(6.4, (0.5 + 0.5*len(self.light_curves)*4.8)))
            
            for i in range(len(self.light_curves)):
                k = list(self.light_curves.keys())[i]
                axs[i].errorbar(np.append(results[k], results[k] + 1),
                                np.append(self.light_curves[k]['relative flux'], self.light_curves[k]['relative flux']),
                                np.append(self.light_curves[k]['relative flux error'], self.light_curves[k]['relative flux error']),
                                marker='.', ms=2, linestyle='none', color=self.colours[k], ecolor='grey', elinewidth=1)
                axs[i].set_title(k)
            
            axs[-1].set_xlabel('Phase')
            axs[len(self.light_curves)//2].set_ylabel('Relative Flux')
        
        return results
    
    def phase_bin(self, period: Quantity, n_bins: int = 10, plot=True) -> Dict[str, Dict[str, NDArray]]:
        """
        Phase bin each light curve using the given period.
        
        Parameters
        ----------
        period : Quantity
            The period to use for phase binning. This must be an astropy `Quantity` with units of time (e.g.,
            `astropy.units.s`) to ensure correct handling of the period.
        n_bins : int, optional
            The number of phase bins, by default 10.
        plot : bool, optional
            Whether to plot the phase binned light curves, by default True.
        
        Returns
        -------
        Dict[str, Dict[str, NDArray]]
            The phase binned light curves.
        """
        
        period = period.to(u.day).value  # convert from given units to days
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            phase = (lc['BDT'] % period) / period
            bins = [[] for i in range(n_bins + 1)]
            
            for i in range(len(phase)):
                bin_num = int(phase[i] * (n_bins + 1))
                bins[bin_num].append(lc['relative flux'][i])
            
            # remove final bin (it will be the same as the first)
            bins.pop()
            
            fluxes = np.array([np.mean(b) for b in bins])
            errs = np.array([np.std(b) / np.sqrt(len(b)) for b in bins])
            
            results[fltr] = {
                'phase': np.linspace(0, 1, n_bins + 1)[:-1],  # remove final phase value (it will be the same as the first)
                'flux': fluxes,
                'flux error': errs
            }
        
        if plot:
            fig, axs = plt.subplots(tight_layout=True, nrows=len(self.light_curves), sharex=True,
                                    figsize=(6.4, (0.5 + 0.5*len(self.light_curves)*4.8)))
            
            for i in range(len(self.light_curves)):
                k = list(self.light_curves.keys())[i]
                axs[i].errorbar(
                    np.append(results[k]['phase'], results[k]['phase'] + 1),
                    np.append(results[k]['flux'], results[k]['flux']),
                    np.append(results[k]['flux error'], results[k]['flux error']),
                    marker='.', ms=2, linestyle='none', color=self.colours[k], ecolor='grey', elinewidth=1)
                axs[i].set_title(k)
            
            axs[-1].set_xlabel('Phase')
            axs[len(self.light_curves)//2].set_ylabel('Relative Flux')
            
            for ax in axs.flatten():
                ax.minorticks_on()
                ax.tick_params(which='both', direction='in', top=True, right=True)
            
            plt.show()
        
        return results
    
    def rebin(self, dt: Quantity, column: Literal['MJD', 'BDT'] = 'BDT') -> 'Analyser':
        """
        Rebin the light curves to a new time resolution. This is useful for smoothing the light curves for presentation
        or for FFT-based analyses.
        
        Parameters
        ----------
        dt : Quantity
            The desired time resolution. This must be an astropy `Quantity` with units of time (e.g., `astropy.units.s`)
            to ensure correct handling of the time resolution.
        column : Literal['MJD', 'BDT'], optional
            The time column to use for rebinned light curves, by default 'MJD'.
        
        Returns
        -------
        Analyser
            A new `Analyser` object containing the rebinned light curves.
        """
        
        def rebin_lc(lc: pd.DataFrame, dt: float, column: Literal['MJD', 'BDT']):
            """
            Rebin a light curve to a specified time resolution.
            
            Parameters
            ----------
            lc : pd.DataFrame
                The light curve to rebin.
            dt : float
                The desired time bin width for rebinned data.
            
            Returns
            -------
            Tuple[NDArray, NDArray, NDArray]
                The rebinned time array, the rebinned data values, and the rebinned error values.
            """
            
            dt = float(dt.to(u.day).value)  # convert from given units to days
            current_dt = np.min(np.diff(lc[column].values))
            minimum_number_of_points = int(dt // current_dt)
            
            new_x = np.arange(lc[column].values[0], lc[column].values[-1] + dt / 2, dt) + dt / 2
            new_y = np.zeros_like(new_x)
            new_yerr = np.zeros_like(new_x)
            
            for i in range(len(new_x)):
                mask = (lc[column].values >= new_x[i] - dt / 2) & (lc[column].values < new_x[i] + dt / 2)
                
                if np.any(mask) and np.sum(mask) >= minimum_number_of_points:
                    new_y[i] = np.mean(lc['relative flux'].values[mask])
                    new_yerr[i] = np.sqrt(np.sum(lc['relative flux error'].values[mask]**2)) / np.sum(mask)
                else:
                    new_y[i] = np.nan
                    new_yerr[i] = np.nan
            
            # drop NaNs
            valid_mask = ~np.isnan(new_y)
            
            rebinned_lc = pd.DataFrame({
                'MJD': new_x[valid_mask],
                'BDT': new_x[valid_mask],
                'relative flux': new_y[valid_mask],
                'relative flux error': new_yerr[valid_mask]
            })
            
            return rebinned_lc
        
        rebinned_light_curves = {}
        
        for fltr, lc in self.light_curves.items():
            rebinned_light_curves[fltr] = rebin_lc(lc.copy(), dt, column)
        
        return Analyser(rebinned_light_curves, self.out_directory, self.prefix, self.phot_type)
    
    def compute_averaged_periodograms(self, dt: Quantity, segment_length: Quantity,
                                      scale: Literal['linear', 'loglog'] = 'loglog',
                                      rebin_frequencies: bool = False, rebin_factor: float = 1.02,
                                      show_plot=True) -> Dict[str, NDArray]:
        """
        Compute the periodogram for each light curve using the FFT. All light curves are assumed to have the same
        time resolution, so it's a good idea to call `Analyser.rebin()` before calling this method.
        
        Parameters
        ----------
        dt : Quantity
            The time resolution of the light curves. This must be an astropy `Quantity` with units of time
            (e.g., `astropy.units.s`) to ensure correct handling of the time resolution.
        segment_length : Quantity
            The length of the segment to use for the periodogram. This must be an astropy `Quantity` with units of time
            (e.g., `astropy.units.s`) to ensure correct handling of the segment length.
        scale : Literal['linear', 'loglog'], optional
            The scale to use for the plot, by default 'loglog'. This does not affect the periodogram frequencies.
        rebin_frequencies : bool, optional
            Whether to rebin the frequencies to a logarithmic scale, by default False. If True, the frequencies will be
            rebinned using `rebin_factor`.
        rebin_factor : float, optional
            The factor by which to rebin the frequencies, by default 1.02. This is only used if `rebin_frequencies` is
            True.
        show_plot : bool, optional
            Whether to show the plot of the periodogram(s), by default True.
        
        Returns
        -------
        Dict[str, NDArray]
            A dictionary containing the periodograms for each light curve.
        """
        
        def rebin_freqs(freqs: NDArray, powers: NDArray, power_errs: NDArray,
                              factor: float) -> Tuple[NDArray, NDArray, NDArray]:
            
            new_freqs = []
            new_powers = []
            new_power_errs = []
            
            prev = 0
            i = 1
            
            bin_width = freqs[0]
            
            while i < len(freqs):
                if freqs[i] - freqs[prev] >= bin_width:
                    new_freqs.append(np.mean(freqs[prev:i]))
                    new_powers.append(np.mean(powers[prev:i]))
                    new_power_errs.append(np.sqrt(np.sum(power_errs[prev:i]**2)) / (i - prev))
                    prev = i
                    bin_width *= factor
                i += 1
            
            return np.array(new_freqs), np.array(new_powers), np.array(new_power_errs)
        
        number_of_points_per_segment = int(segment_length // dt)
        frequencies = np.fft.rfftfreq(number_of_points_per_segment, dt)[1:]  # skip the zero frequency
        segment_length = segment_length.to(u.day).value  # convert from given units to days
        
        segment_edges = np.arange(self.light_curves[list(self.light_curves.keys())[0]]['BDT'].min(),
                                  self.light_curves[list(self.light_curves.keys())[0]]['BDT'].max() + segment_length,
                                  segment_length)
        results = {}
        
        for fltr, lc in self.light_curves.items():
            segment_powers = []
            results[fltr] = {}
            prev = 0.
            
            for edge in segment_edges[1:]:
                mask = (lc['BDT'].values >= prev) & (lc['BDT'].values < edge)
                if np.sum(mask) == number_of_points_per_segment:
                    segment_powers.append(np.abs(np.fft.rfft(lc['relative flux'].values[mask])[1:])**2)
                prev = edge
            
            n_segments = len(segment_powers)
            if n_segments < 10:
                warnings.warn(f'[OPTICAM] Only {n_segments} segments were found for filter {fltr}. Consider reducing segment_length to increase the number of segments.')
            
            results[fltr]['powers'] = np.mean(segment_powers, axis=0)
            results[fltr]['power_errs'] = np.std(segment_powers, axis=0) / np.sqrt(len(segment_powers))
            
            if rebin_frequencies:
                rebinned_freqs, rebinned_powers, rebinned_power_errs = rebin_freqs(frequencies,
                                                                                   results[fltr]['powers'],
                                                                                   results[fltr]['power_errs'],
                                                                                   rebin_factor)
                results[fltr]['powers'] = rebinned_powers
                results[fltr]['power_errs'] = rebinned_power_errs
        
        if rebin_frequencies:
            return rebinned_freqs, results
        else:
            return frequencies, results