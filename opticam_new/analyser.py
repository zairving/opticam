import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from typing import Dict, Union, Literal
from numpy.typing import ArrayLike, NDArray
import os


class Analyser:
    """
    Helper class for analysing OPTICam light curves.
    """
    
    def __init__(self, light_curves: Dict[str, pd.DataFrame], out_directory: str, prefix: str, phot_type: str):
        """
        Helper class for analysing OPTICam light curves.

        Parameters
        ----------
        light_curves : Dict[str, pd.DataFrame]
            The light curves to analyse, where the keys are the filter names and the values are the light curves.
        out_directory : str
            The directory to save the output files.
        prefix : str
            The prefix to use for the output files.
        phot_type : str
            The type of photometry used to generate the light curves.
        """
        
        self.light_curves = light_curves
        self.filters = list(light_curves.keys())
        self.out_directory = out_directory
        self.prefix = prefix
        self.phot_type = phot_type
        
        self.colours = {
            'g-band': 'green',  # camera 1
            'u-band': 'green',  # camera 1
            'r-band': 'red',  # camera 2
            'i-band': 'gold',  # camera 3
            'z-band': 'gold',  # camera 3
        }
    
    def plot(self, title: str = None, x_col: Literal['MJD', 'BDT'] = 'BDT', ax = None):
        """
        Plot the light curves.
        
        Parameters
        ----------
        title : str, optional
            _description_, by default None
        x_col : Literal[&#39;MJD&#39;, &#39;BDT&#39;], optional
            _description_, by default 'BDT'
        ax : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        
        if ax is None:
            fig, ax = plt.subplots()
        
        for fltr, lc in self.light_curves.items():
            ax.errorbar(lc[x_col], lc['relative flux'], lc['relative flux error'], marker='.', ms=2,
                        linestyle='none', color=self.colours[fltr], ecolor='grey', elinewidth=1, label=fltr)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel('Relative Flux')
        
        if title is not None:
            ax.set_title(title)
        
        ax.legend()
        
        plt.show()
        
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
        Combine the light curves of another analyser with the current one(s).
        
        Parameters
        ----------
        analyser : Analyser
            The analyser object whose light curves are to be combined with the current one(s).
        """
        
        light_curves = analyser.light_curves
        filters = analyser.filters
        
        for fltr in filters:
            self.light_curves[fltr] = light_curves[fltr].copy()
    
    def lomb_scargle(self, frequencies: NDArray = None, scale: Literal['linear', 'log', 'loglog'] = 'linear', 
                     show_plot=True) -> Union[Dict[str, NDArray], NDArray, Dict[str, NDArray]]:
        """
        Compute the Lomb-Scargle periodogram for each light curve.
        
        Parameters
        ----------
        frequencies : NDArray, optional
            The periodogram frequencies, by default None. If None, a suitable frequency range will be inferred from the
            data.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the inferred frequencies, by default 'linear'. If 'linear', numpy.linspace will be used
            to generate the frequencies. If 'log', numpy.logspace will be used, and the frequency axis will be in
            logarithm. The upper and lower bounds of the frequencies are always the same. If 'loglog', both the
            frequency and power axes will be in logarithm.
        show_plot : bool, optional
            Whether to show the plot of the periodogram(s), by default True.
        Returns
        -------
        Union[Dict[str, NDArray], NDArray, Dict[str, NDArray]]
            The Lomb-Scargle periodogram for each light curve. If the frequencies are not provided, the inferred
            frequencies are also returned.
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
                                figsize=(6.4, (0.5 + 0.5*len(self.light_curves)*4.8)))
        
        for i in range(len(self.light_curves)):
            
            k = list(self.light_curves.keys())[i]
            power = LombScargle((self.light_curves[k]['BDT'].values - self.light_curves[k]['BDT'].min())*86400,
                                self.light_curves[k]['relative flux'],
                                self.light_curves[k]['relative flux error']).power(frequencies)
            results[k] = power
            
            axs[i].plot(frequencies, power, color=self.colours[k], lw=1, label=k)
            axs[i].set_title(k)
        
        if scale == 'log':
            for ax in axs:
                ax.set_xscale('log')
        
        if scale == 'loglog':
            for ax in axs:
                ax.set_xscale('log')
                ax.set_yscale('log')
        
        axs[-1].set_xlabel('Frequency (Hz)')
        axs[len(self.light_curves)//2].set_ylabel('Power')
        
        fig.savefig(f'{self.out_directory}/periodograms/{self.prefix}_{self.phot_type}_periodogram.png')
        
        if show_plot:
            plt.show()
        
        if return_frequencies:
            return frequencies, results
        
        return results
    
    def phase_fold(self, period: float) -> Dict[str, NDArray]:
        """
        Phase fold each light curve using the given period.
        
        Parameters
        ----------
        period : float
            The period to use for phase folding.
        
        Returns
        -------
        Dict[str, NDArray]
            The phase folded light curves.
        """
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            phase = (lc['BDT'] % period) / period
            results[fltr] = phase
        
        return results
    
    def phase_bin(self, period: np.array, n_bins: int = 10):
        """
        Phase bin the light curve.

        Parameters
        ----------
        period : np.array
            The period used to fold the light curve.
        n_bins : int, optional
            The number of phase bins. The default is 100.

        Returns
        -------
        """
        
        bins = np.linspace(0, 1, n_bins + 1)[1:]  # remove first element since phase cannot be less than zero (resulting in a bin with no points)
        
        phases = self.phase_fold(period)
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
        
            digitized = np.digitize(phases[fltr], bins)
            
            lengths = np.array([len(lc['relative flux'][digitized == i]) for i in range(len(bins))], np.float64).flatten()
            
            # if bin has fewer than 30 points, reduce number of bins
            if np.any(np.array(lengths) < 30):
                # reduce number of bins until each bin has at least 30 points
                return self.phase_bin(period, n_bins - 1)
            else:
                folded_y = np.array([np.mean(self.f[digitized == i]) for i in range(len(bins))], np.float64).flatten()
                folded_yerr = np.array([np.sqrt(np.sum(np.square(self.ferr[digitized == i])))/len(self.ferr[digitized == i]) for i in range(len(bins))], np.float64).flatten()
            
            return bins - bins[0]/2, folded_y, folded_yerr