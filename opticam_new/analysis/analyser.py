import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Literal, Tuple
from numpy.typing import NDArray
import os
import astropy.units as u
from astropy.units.quantity import Quantity
from stingray import Lightcurve
from matplotlib.figure import Figure
import copy
from stingray import AveragedPowerspectrum, Powerspectrum
from stingray import AveragedCrossspectrum, Crossspectrum
from stingray.lombscargle import LombScarglePowerspectrum
from stingray import CrossCorrelation
from pandas import DataFrame

from opticam_new.utils.helpers import sort_filters
from opticam_new.utils.time_helpers import infer_gtis
from opticam_new.utils.constants import colors

class Analyser:
    """
    Helper class for analysing OPTICAM light curves.
    """

    def __init__(
        self,
        out_directory: str,
        light_curves: Dict[str, Lightcurve | DataFrame] | None = None,
        prefix: str | None = None,
        phot_label: str | None = None,
        show_plots: bool = True,
        ) -> None:
        """
        Helper class for analysing OPTICAM light curves.
        
        Parameters
        ----------
        out_directory : str
            The directory to save the output files (i.e., the same directory as `out_directory` used by
            `opticam_new.Photometer` when creating the light curves).
        light_curves : Dict[str, Lightcurve | DataFrame] | None, optional
            The light curves to analyse, where the keys are the filter names and the values are either Lightcurve
            objects or DataFrames containing 'TDB', 'rel_flux', and 'rel_flux_err' columns. Leave as `None` to create an
            empty analyser that can be populated later using the `join()` method.
        prefix : str | None, optional
            The prefix to use for the output files (e.g., the name of the target source).
        phot_label : str, optional
            The label for the photometry routine used to generate the light curves, used in the output file names.
        show_plots : bool, optional
            Whether to render and show plots, by default `True`.
        """
        
        self.light_curves = self._validate_light_curves(light_curves)
        
        self.out_directory = out_directory
        if not os.path.isdir(out_directory):
            try:
                os.makedirs(out_directory)
            except Exception as e:
                raise Exception(f'[OPTICAM] could not create output directory {out_directory}: {e}')
        
        self.prefix = prefix
        self.phot_label = phot_label
        self.show_plots = show_plots
        
        if self.show_plots:
            if not os.path.isdir(os.path.join(self.out_directory, 'plots')):
                os.makedirs(os.path.join(self.out_directory, 'plots'))
        
        if len(self.light_curves) > 0:
            self.t_ref = float(min([np.min(lc.time) for lc in self.light_curves.values()]))

    @staticmethod
    def _validate_light_curves(
        light_curves: Dict[str, Lightcurve | DataFrame] | None,
        ) -> Dict[str, Lightcurve]:
        """
        Validate the light curves by converting DataFrames to Lightcurve objects and inferring GTIs.
        
        Parameters
        ----------
        light_curves : Dict[str, Lightcurve | DataFrame] | None
            The light curves to validate, where the keys are the filter names and the values are either Lightcurve
            objects or DataFrames containing 'TDB', 'rel_flux', and 'rel_flux_err' columns. If `None`, an empty
            dictionary will be returned.
        
        Returns
        -------
        Dict[str, Lightcurve]
            If `light_curves` is `None`, returns an empty dictionary. Otherwise, returns a dictionary containing the 
            validated light curves, where the keys are the filter names and the values are Lightcurve objects.
        """
        
        validated_light_curves = {}
        
        if light_curves:
            for fltr in light_curves.keys():
                if isinstance(light_curves[fltr], DataFrame):
                    time = np.asarray(light_curves[fltr]['TDB'].values)
                    counts = np.asarray(light_curves[fltr]['rel_flux'].values)
                    counts_err = np.asarray(light_curves[fltr]['rel_flux_err'].values)
                    
                    # infer GTIs
                    gtis = infer_gtis(time, threshold=1.5)
                elif isinstance(light_curves[fltr], Lightcurve):
                    time = np.asarray(light_curves[fltr].time)
                    counts = np.asarray(light_curves[fltr].counts)
                    counts_err = np.asarray(light_curves[fltr].counts_err)
                    gtis = light_curves[fltr].gti
                else:
                    raise TypeError(f'[OPTICAM] Light curve for filter {fltr} must be either a DataFrame or a Lightcurve object, but got {type(light_curves[fltr])}.')
                
                # normalise flux
                mean_flux = np.mean(counts)
                counts /= mean_flux
                counts_err /= mean_flux
                
                # convert DataFrame to Lightcurve
                validated_light_curves[fltr] = Lightcurve(
                    time,
                    counts,
                    err=counts_err,
                    gti=gtis,
                    err_dist='gauss',
                    )
        
        return sort_filters(validated_light_curves)


    def join(
        self,
        analyser: 'Analyser',
        ) -> 'Analyser':
        """
        Combine another `Analyser` instance with the current one. If the new `Analyser` has light curves with filters
        that are not present in the current `Analyser`, those filters will be added. If the new `Analyser` has light
        curves with filters that are already present in the current `Analyser`, those light curves will be merged.
        
        Parameters
        ----------
        analyser : Analyser
            The analyser instance being combined with the current one.
        
        Returns
        -------
        Analyser
            A new `Analyser` instance with the combined light curves.
        """
        
        assert analyser.light_curves, f'[OPTICAM] cannot join an empty analyser.'
        
        new_light_curves = copy.copy(self.light_curves)
        
        for fltr in analyser.light_curves.keys():
            if fltr not in self.light_curves.keys():
                # if a new filter is being added, copy the light curve
                new_light_curves[fltr] = copy.copy(analyser.light_curves[fltr])
            else:
                # if an existing filter is being added, merge the light curves
                new_light_curves[fltr] = self.light_curves[fltr].join(analyser.light_curves[fltr])
        
        return Analyser(
            out_directory=self.out_directory,
            light_curves=new_light_curves,
            prefix=self.prefix,
            phot_label=self.phot_label,
            show_plots=self.show_plots,
        )

    def rebin_light_curves(
        self,
        dt: Quantity,
        ) -> None:
        """
        Rebin the light curves to a desired time resolution using `stingray.Lightcurve.rebin()`.
        
        Parameters
        ----------
        dt : Quantity
            The desired time resolution for the rebinned light curves. This must be an astropy `Quantity` with units of
            time (e.g., `astropy.units.s`) to ensure correct handling of the time resolution.
        """
        
        # convert dt to days
        dt = dt.to(u.day).value
        
        for fltr, lc in self.light_curves.items():
            self.light_curves[fltr] = lc.rebin(dt, method='mean')

    def _convert_lc_time_to_seconds(
        self,
        lc: Lightcurve,
        ) -> Lightcurve:
        """
        Convert the time of a light curve from days to seconds, relative to the reference time.
        
        Parameters
        ----------
        lc : Lightcurve
            The light curve to convert.
        
        Returns
        -------
        Lightcurve
            The light curve with time converted to seconds, relative to the reference time.
        """
        
        t = (np.asarray(lc.time) - self.t_ref) * 86400
        gti = (np.asarray(lc.gti) - self.t_ref) * 86400
        
        return Lightcurve(
            t,
            lc.counts,
            err=lc.counts_err,
            gti=gti,
            err_dist=lc.err_dist,
        )


    def plot_light_curves(
        self,
        title: str | None = None,
        ) -> Figure:
        """
        Plot the light curves.
        
        Parameters
        ----------
        title : str | None, optional
            The figure title, by default `None`.
        
        Returns
        -------
        Figure
            The figure containing the light curves.
        """
        
        fig, axes = plt.subplots(
            nrows=len(self.light_curves),
            figsize=(2 * 6.4, .5 * len(self.light_curves) * 4.8),
            tight_layout=True,
            sharex=True,
            gridspec_kw={
                "hspace": 0,
                },
            )
        
        if len(self.light_curves) == 1:
            axes = [axes]
        
        for i, (fltr, lc) in enumerate(self.light_curves.items()):
            
            for lc_segment in lc.split_by_gti(min_points=1):
            
                t = (np.asarray(lc_segment.time) - self.t_ref) * 86400
                
                axes[i].errorbar(
                    t,
                    lc_segment.counts,
                    lc_segment.counts_err,
                    marker='none',
                    linestyle='none',
                    ecolor='grey',
                    elinewidth=1,
                    )
                axes[i].step(
                    t,
                    lc_segment.counts,
                    where='mid',
                    color=colors[fltr],
                    lw=1,
                    label=fltr,
                )
            
            axes[i].text(
                .95,
                .9,
                fltr,
                fontsize='large',
                transform=axes[i].transAxes,
                va='top',
                ha='right',
            )
        
        axes[-1].set_xlabel(f'Time from BMJD {self.t_ref:.4f} [s]', fontsize='large')
        axes[len(self.light_curves) // 2].set_ylabel('Normalised flux', fontsize='large')
        
        if title is not None:
            axes[0].set_title(title)
        
        for ax in axes:
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', top=True, right=True)
        
        fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_light_curves.png')
        
        if self.show_plots:
            plt.show()
        
        return fig


    def phase_fold_light_curves(
        self,
        period: Quantity,
        ) -> Dict[str, NDArray]:
        """
        Phase fold each light curve using the given period.
        
        Parameters
        ----------
        period : Quantity
            The period to use for phase folding. This must be an astropy `Quantity` with units of time (e.g.,
            `astropy.units.s`) to ensure correct handling of the period.
        
        Returns
        -------
        Dict[str, NDArray]
            The phase folded light curves.
        """
        
        period = period.to(u.day).value  # convert from given units to days
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            phase = (lc.time % period) / period
            results[fltr] = phase
        
        if self.show_plots:
            fig, axs = plt.subplots(tight_layout=True, nrows=len(self.light_curves), sharex=True,
                                    figsize=(6.4, (0.5 + 0.5*len(self.light_curves)*4.8)))
            
            for i in range(len(self.light_curves)):
                k = list(self.light_curves.keys())[i]
                axs[i].errorbar(np.append(results[k], results[k] + 1),
                                np.append(self.light_curves[k].counts, self.light_curves[k].counts),
                                np.append(self.light_curves[k].counts_err, self.light_curves[k].counts_err),
                                marker='.', ms=2, linestyle='none', color=colors[k], ecolor='grey', elinewidth=1)
                axs[i].set_title(k)
            
            axs[-1].set_xlabel('Phase')
            axs[len(self.light_curves)//2].set_ylabel('Relative Flux')
        
        return results

    def phase_bin_light_curves(
        self,
        period: Quantity,
        t0: float | None = None,
        n_bins: int = 10,
        plot=True,
        ) -> Dict[str, Dict[str, NDArray]]:
        """
        Phase bin each light curve using the given period.
        
        Parameters
        ----------
        period : Quantity
            The period to use for phase binning. This must be an astropy `Quantity` with units of time (e.g.,
            `astropy.units.s`) to ensure correct handling of the period.
        t0 : float | None, optional
            Time of zero phase, by default `None`. If `None`, the first time value in the light curve will be used.
        n_bins : int, optional
            The number of phase bins, by default 10.
        plot : bool, optional
            Whether to plot the phase binned light curves, by default True.
        
        Returns
        -------
        Dict[str, Dict[str, NDArray]]
            The phase binned light curves.
        """
        
        period = period.to_value(u.day)  # convert from given units to days
        
        results = {}
        
        for fltr, lc in self.light_curves.items():
            
            lc_time = np.asarray(lc.time)
            
            if t0:
                t = lc_time - t0
            else:
                t = lc_time - lc_time[0]
            
            phase = (t % period) / period
            bins = [[] for i in range(n_bins + 1)]
            bin_errs = [[] for i in range(n_bins + 1)]
            
            for i in range(len(phase)):
                bin_num = int(phase[i] * (n_bins + 1))
                bins[bin_num].append(np.asarray(lc.counts)[i])
                bin_errs[bin_num].append(np.asarray(lc.counts_err)[i])
            
            # remove final bin (it will be the same as the first)
            bins.pop()
            bin_errs.pop()
            
            fluxes = np.array([np.mean(b) for b in bins])
            flux_errs = np.array(np.sqrt([np.sum(np.asarray(b)**2) for b in bin_errs]) / len(bins[0]))
            
            results[fltr] = {
                'phase': np.linspace(0, 1, n_bins + 1)[:-1],  # remove final phase value (it will be the same as the first)
                'flux': fluxes,
                'flux error': flux_errs
            }
        
        if plot:
            fig, axes = plt.subplots(
                tight_layout=True,
                nrows=len(self.light_curves),
                sharex=True,
                figsize=(6.4, (0.5 * len(self.light_curves) * 4.8)),
                gridspec_kw={'hspace': 0.},
                )
            
            for i in range(len(self.light_curves)):
                k = list(self.light_curves.keys())[i]
                axes[i].errorbar(
                    np.append(results[k]['phase'], results[k]['phase'] + 1),
                    np.append(results[k]['flux'], results[k]['flux']),
                    np.append(results[k]['flux error'], results[k]['flux error']),
                    marker='none', linestyle='none', color=colors[k], ecolor='grey', elinewidth=1)
                axes[i].step(
                    np.append(results[k]['phase'], results[k]['phase'] + 1),
                    np.append(results[k]['flux'], results[k]['flux']),
                    where='mid',
                    color=colors[k],
                    lw=1,
                )
                axes[i].text(
                    .95,
                    .9,
                    k,
                    fontsize='large',
                    va='top',
                    ha='right',
                    transform=axes[i].transAxes,
                )
            
            axes[-1].set_xlabel('Phase')
            axes[len(self.light_curves)//2].set_ylabel('Normalised flux')
            
            for ax in axes.flatten():
                ax.minorticks_on()
                ax.tick_params(which='both', direction='in', top=True, right=True)
            
            plt.show()
        
        return results


    def compute_power_spectra(
        self,
        norm: Literal['frac', 'abs'] = 'frac',
        scale: Literal['linear', 'log', 'loglog'] = 'linear',
        ) -> Dict[str, Powerspectrum]:
        """
        Compute the power spectrum for each light curve using `stingray.Powerspectrum`. It's usually a good idea to call 
        the rebin() method to rebin your light curves to a regular time grid before calling this method.
        
        Parameters
        ----------
        norm : Literal['frac', 'abs'], optional
            The normalisation to use for the power spectrum, by default 'frac'. If 'frac', the power spectrum is
            normalised to fractional rms. If 'abs', the power spectrum is normalised to absolute power.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the plot, by default 'linear'. If 'linear', all axes are linear. If 'log', the
            frequency axis is logarithmic. If 'loglog', both the frequency and power axes are logarithmic.
        
        Returns
        -------
        Dict[str, Powerspectrum]
            A dictionary containing the power spectrum for each light curve, where the keys are the filter names and the
            values are the power spectra.
        """
        
        results = {}
        
        for fltr in self.light_curves.keys():
            
            lc = self._convert_lc_time_to_seconds(self.light_curves[fltr])
            
            ps = Powerspectrum.from_lightcurve(
                lc,
                norm=norm,
                silent=True,
                )
            
            results[fltr] = ps
        
        if self.show_plots:
            fig = _plot(results, scale)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_periodograms.png')
            plt.show()
        
        return results

    def compute_averaged_power_spectra(
        self,
        segment_size: Quantity,
        rebin_factor: float | None = None,
        norm: Literal['frac', 'abs'] = 'frac',
        scale: Literal['linear', 'log', 'loglog'] = 'linear',
        ) -> Dict[str, AveragedPowerspectrum]:
        """
        Compute the averaged power spectrum for each light curve using `stingray.AveragedPowerSpectrum`. It's usually a
        good idea to call the rebin() method to rebin your light curves to a regular time grid before calling this 
        method.
        
        Parameters
        ----------
        segment_size : Quantity
            The size of the segments to use for averaging the power spectra. This must be an astropy `Quantity` with
            units of time (e.g., `astropy.units.s`) to ensure correct handling of the segment size.
        rebin_factor : float | None, optional
            The factor by which to rebin the power spectrum in frequency. If 'None', no rebinning will be performed.
            If a float, the power spectrum will be geometrically/logarithmically rebinned with each bin being a factor
            `1 + rebin_factor` larger than the previous one.
        norm : Literal['frac', 'abs'], optional
            The normalisation to use for the power spectrum, by default 'frac'. If 'frac', the power spectrum is
            normalised to the fractional rms. If 'abs', the power spectrum is normalised to the absolute rms.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the plot, by default 'linear'. If 'linear', all axes are linear. If 'log', the
            frequency axis is logarithmic. If 'loglog', both the frequency and power axes are logarithmic.
        
        Returns
        -------
        Dict[str, AveragedPowerspectrum]
            The averaged power spectrum for each light curve, where the keys are the filter names and the values are
            the averaged power spectra.
        """
        
        segment_size = segment_size.to(u.s).value  # convert from given units to days
        
        results = {}
        for k in self.light_curves.keys():
            
            lc = self._convert_lc_time_to_seconds(self.light_curves[k])
            
            ps = AveragedPowerspectrum.from_lightcurve(
                lc,
                segment_size,
                norm=norm,
                silent=True,
            )
            
            print(f'[OPTICAM] {ps.m} {k} segments averaged.')
            
            if rebin_factor:
                ps = ps.rebin_log(rebin_factor)
            
            results[k] = ps
        
        if self.show_plots:
            fig = _plot(results, scale)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_averaged_power_spectra.png')
            plt.show()
        
        return results


    def compute_crossspectra(
        self,
        norm: Literal['frac', 'abs'] = 'frac',
        scale: Literal['linear', 'log', 'loglog'] = 'linear',
        ) -> Dict[str, Crossspectrum]:
        """
        Compute the cross-spectra for each pair of light curves using `stingray.Crossspectrum`. It's usually a good idea
        to call the rebin() method to rebin your light curves to a regular time grid before calling this method.
        
        Parameters
        ----------
        norm : Literal['frac', 'abs'], optional
            The normalisation to use for the cross-spectrum, by default 'frac'. If 'frac', the cross-spectrum is
            normalised to fractional rms. If 'abs', the cross-spectrum is normalised to absolute power.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the plot, by default 'linear'. If 'linear', all axes are linear. If 'log', the
            frequency axis is logarithmic. If 'loglog', both the frequency and power axes are logarithmic.
        
        Returns
        -------
        Dict[str, Crossspectrum]
            A dictionary containing the cross-spectra for each pair of light curves, where the keys are tuples of
            filter names and the values are the cross-spectra.
        """
        
        results = {}
        
        for fltr1, lc1 in self.light_curves.items():
            for fltr2, lc2 in self.light_curves.items():
                if fltr1 == fltr2:
                    continue
                
                # convert light curves from days to seconds
                lc1 = self._convert_lc_time_to_seconds(lc1)
                lc2 = self._convert_lc_time_to_seconds(lc2)
                
                cs = Crossspectrum.from_lightcurve(
                    lc1,
                    lc2,
                    norm=norm,
                    silent=True,
                    )
                results[(fltr1, fltr2)] = cs
        
        if self.show_plots:
            fig = _plot(results, scale)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_cross_spectra.png')
            plt.show()
        
        return results

    def compute_averaged_crossspectra(
        self,
        segment_size: Quantity,
        norm: Literal['frac', 'abs'] = 'frac',
        scale: Literal['linear', 'log', 'loglog'] = 'linear',
        ) -> Dict[str, AveragedCrossspectrum]:
        """
        Compute the cross-spectra for each pair of light curves using `stingray.Crossspectrum`. It's usually a good idea
        to call the rebin() method to rebin your light curves to a regular time grid before calling this method.
        
        Parameters
        ----------
        segment_size : Quantity
            The size of the segments to use for averaging the cross-spectra. This must be an astropy `Quantity` with
            units of time (e.g., `astropy.units.s`) to ensure correct handling of the segment size.
        norm : Literal['frac', 'abs'], optional
            The normalisation to use for the cross-spectrum, by default 'frac'. If 'frac', the cross-spectrum is
            normalised to fractional rms. If 'abs', the cross-spectrum is normalised to absolute power.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the plot, by default 'linear'. If 'linear', all axes are linear. If 'log', the
            frequency axis is logarithmic. If 'loglog', both the frequency and power axes are logarithmic.
        
        Returns
        -------
        Dict[str, AveragedCrossspectrum]
            A dictionary containing the averaged cross-spectra for each pair of light curves, where the keys are tuples
            of filter names and the values are the cross-spectra.
        """
        
        segment_size = segment_size.to(u.day).value  # convert from given units to days
        
        results = {}
        
        for fltr1, lc1 in self.light_curves.items():
            for fltr2, lc2 in self.light_curves.items():
                if fltr1 == fltr2:
                    continue
                
                # convert light curves from days to seconds
                lc1 = self._convert_lc_time_to_seconds(lc1)
                lc2 = self._convert_lc_time_to_seconds(lc2)
                
                cs = AveragedCrossspectrum.from_lightcurve(
                    lc1,
                    lc2,
                    segment_size=segment_size,
                    norm=norm,
                    silent=True,
                    )
                
                print(f'[OPTICAM] {cs.m} {fltr1} x {fltr2} segments averaged.')
                
                results[(fltr1, fltr2)] = cs
        
        if self.show_plots:
            fig = _plot(results, scale)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_averaged_cross_spectra.png')
            plt.show()
        
        return results


    def compute_lomb_scargle_periodograms(
        self,
        norm: Literal['abs', 'frac'] = 'frac',
        scale: Literal['linear', 'log', 'loglog'] = 'linear',
        ) -> Dict[str, LombScarglePowerspectrum]:
        """
        Compute the Lomb-Scargle periodogram for each light curve using `stingray.LombScarglePowerspectrum`.
        
        Parameters
        ----------
        norm : Literal['abs', 'frac'], optional
            The normalisation to use for the Lomb-Scargle periodogram, by default 'frac'. If 'abs', the periodogram is
            normalised to absolute power. If 'frac', the periodogram is normalised to fractional rms.
        scale : Literal['linear', 'log', 'loglog'], optional
            The scale to use for the inferred frequencies, by default 'linear'. If 'linear', the frequency grid is
            linearly spaced. If 'log', the frequency grid is logarithmically spaced. If 'loglog', both the frequency
            and power axes will be in logarithm. The upper and lower bounds of the frequencies are the same in all
            cases.
        Returns
        -------
        Tuple[NDArray, Dict[str, NDArray]] | Dict[str, NDArray]
            If no frequencies are provided, returns a tuple containing the frequencies and a dictionary of periodograms
            for each light curve. If frequencies are provided, returns a dictionary of periodogram powers for each light
            curve.
        """
        
        results = {}
        
        for k in self.light_curves.keys():
            
            lc = self._convert_lc_time_to_seconds(self.light_curves[k])
            
            # don't use .from_lightcurve() here as it causes an error
            lsp = LombScarglePowerspectrum(
                lc,
                norm=norm,
                power_type='absolute',
            )
            results[k] = lsp
        
        if self.show_plots:
            fig = _plot(results, scale)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_L-S_periodograms.png')
            plt.show()
        
        return results


    def compute_cross_correlations(
        self,
        mode: Literal['same', 'valid', 'full'] = 'same',
        norm: Literal['none', 'variance'] = 'variance',
        force_match: bool = True,
        ) -> Dict[str, CrossCorrelation]:
        """
        Compute the cross-correlations for each pair of light curves using `stingray.CrossCorrelation`.
        
        Parameters
        ----------
        mode : Literal['same', 'valid', 'full'], optional
            The mode to use for the cross-correlation, by default 'same'. See `stingray.CrossCorrelation` for details on
            the different modes.
        norm : Literal['none', 'variance'], optional
            The normalisation to use for the cross-correlation, by default 'variance'. See `stingray.CrossCorrelation`
            for details on the different normalisations.
        force_match : bool, optional
            Whether to force the light curves to have the same time columns before computing the cross-correlation,
            by default `True`. If `False`, cross-correlation calculations may fail if the light curves have different
            time columns.
        
        Returns
        -------
        Dict[str, CrossCorrelation]
            A dictionary containing the cross-correlations for each pair of light curves, where the keys are tuples of
            filter names and the values are the cross-correlations.
        """
        
        results = {}
        
        for fltr1, lc1 in self.light_curves.items():
            for fltr2, lc2 in self.light_curves.items():
                # skip if the filters are the same or if the cross-correlation has already been computed
                if fltr1 == fltr2 or (fltr2, fltr1) in results.keys():
                    continue
                
                if force_match:
                    # force the light curves to have the same time columns
                    lc1, lc2 = _match_light_curve_times(lc1, lc2)
                
                cc = CrossCorrelation(
                    lc1,
                    lc2,
                    mode=mode,
                    norm=norm,
                    )
                # convert times to seconds
                cc.dt *= 86400
                cc.time_lags *= 86400
                cc.time_shift *= 86400
                results[(fltr1, fltr2)] = cc
        
        if self.show_plots:
            fig = _plot(results)
            fig.savefig(f'{self.out_directory}/plots/{self.prefix}_{self.phot_label}_cross_correlations.png')
            plt.show()
        
        return results




def _plot(
    results: Dict[str,
                  AveragedPowerspectrum | 
                  Powerspectrum | 
                  Crossspectrum | 
                  AveragedCrossspectrum | 
                  LombScarglePowerspectrum |
                  CrossCorrelation
                  ],
    scale: Literal['linear', 'log', 'loglog'] = 'linear',
    ) -> Figure:
    """
    Plot the results of some timing analysis. Not intended to be called directly by the user, but rather via
    various timing methods.
    
    Parameters
    ----------
    results : Dict[str, AveragedPowerspectrum  |  Powerspectrum  |  Crossspectrum  |  AveragedCrossspectrum  |  
    LombScarglePowerspectrum | CrossCorrelation]
        The timming analysis results, where the keys are the filter names and the values are the results.
    scale : Literal['linear', 'log', 'loglog'], optional
        The scale to use for the plot, by default 'linear'. If 'linear', all axes are linear. If 'log', the
        x-axis is logarithmic. If 'loglog', both the x- and y-axes are logarithmic.
    
    Returns
    -------
    Figure
        The figure containing the plot.
    """
    
    fig, axes = plt.subplots(
        figsize=(6.4, 4.8 * .5 * len(results)),
        tight_layout=True,
        nrows=len(results),
        sharex=True,
        gridspec_kw={'hspace': 0.},
    )
    
    if len(results) == 1:
        axes = [axes]
    
    for i, key in enumerate(results.keys()):
        if isinstance(results[key], 
                      AveragedPowerspectrum | 
                      Powerspectrum | 
                      Crossspectrum | 
                      AveragedCrossspectrum |
                      LombScarglePowerspectrum
                      ):
            x = results[key].freq
            y = results[key].power
            yerr = _define_yerr(results[key])
            
            x_label = 'Frequency [Hz]'
            y_label = f'Power [{_get_normalisation_units(results[key].norm)}]'
        elif isinstance(results[key], CrossCorrelation):
            x = results[key].time_lags
            y = results[key].corr
            yerr = None
            
            x_label = 'Time lag [s]'
            y_label = 'Correlation'
        
        axes[i].step(
            x,
            y,
            color='k',
            lw=1,
            where='mid',
        )
        
        if yerr is not None:
            axes[i].errorbar(
                x,
                y,
                yerr,
                linestyle='none',
                marker='none',
                ecolor='grey',
                elinewidth=1,
                zorder=0,
                )
        
        axes[i].text(
            .95,
            .9,
            key,
            transform=axes[i].transAxes,
            fontsize='large',
            ha='right',
            va='top',
        )
        
        if isinstance(results[key], CrossCorrelation):
            axes[i].text(
                .05,
                .9,
                f'time shift: {results[key].time_shift:.1f} s',
                transform=axes[i].transAxes,
                fontsize='large',
                ha='left',
                va='top',
            )
    
    if scale == 'log':
        for ax in axes:
            ax.set_xscale('log')
    elif scale == 'loglog':
        for ax in axes:
            ax.set_xscale('log')
            ax.set_yscale('log')
    
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    
    axes[-1].set_xlabel(x_label)
    axes[len(results) // 2].set_ylabel(y_label)
    
    return fig

def _get_normalisation_units(
    norm: str,
    ) -> str:
    """
    Get the units of the normalisation based on the specified normalisation type.
    
    Parameters
    ----------
    norm : str
        The normalisation type.
    
    Returns
    -------
    str
        The units of the normalisation.
    
    Raises
    ------
    ValueError
        If the specified normalisation type is invalid.
    """
    
    return '(fractional rms)$^2$ Hz$^{{-1}}$'

def _define_yerr(
    obj: AveragedPowerspectrum |
    Powerspectrum |
    Crossspectrum |
    AveragedCrossspectrum |
    LombScarglePowerspectrum,
    ) -> NDArray | None:
    """
    Determine the y-error for the given object.
    
    Parameters
    ----------
    obj : AveragedPowerspectrum | Powerspectrum | Crossspectrum | AveragedCrossspectrum | LombScarglePowerspectrum
        The object for which to determine the y-error.
    
    Returns
    -------
    NDArray | None
        The y-error array if it exists (and a sufficient number of segments has been averaged), otherwise `None`.
    """
    
    if isinstance(obj, (AveragedPowerspectrum, AveragedCrossspectrum)):
        return obj.power_err
    
    return None




def _intersect_gtis(gti1, gti2):

    result = []
    for start1, stop1 in gti1:
        for start2, stop2 in gti2:
            start = max(start1, start2)
            stop = min(stop1, stop2)
            if start < stop:
                result.append([start, stop])
    
    return np.array(result)

def _restrict_to_gti(lc, gti):
    
    mask = np.zeros_like(lc.time, dtype=bool)
    for start, stop in gti:
        mask |= (lc.time >= start) & (lc.time <= stop)
        
    return Lightcurve(
        lc.time[mask],
        lc.counts[mask],
        err=lc.counts_err[mask],
        gti=gti,
        err_dist=lc.err_dist,
    )

def _match_light_curve_times(
    lc1: Lightcurve,
    lc2: Lightcurve,
    ) -> Tuple[Lightcurve, Lightcurve]:
    """
    Match the time columns of two light curves.
    
    Parameters
    ----------
    lc1 : Lightcurve
        The first light curve.
    lc2 : Lightcurve
        The second light curve.
    
    Returns
    -------
    Tuple[Lightcurve, Lightcurve]
        The two light curves with matched time columns.
    """
    
    # get intersecting GTIs
    gti = _intersect_gtis(lc1.gti, lc2.gti)
    lc1_restricted = _restrict_to_gti(lc1, gti)
    lc2_restricted = _restrict_to_gti(lc2, gti)
    
    # interpolate lc2 onto lc1's time grid
    interp_counts = np.interp(
        np.asarray(lc1_restricted.time),
        np.asarray(lc2_restricted.time),
        np.asarray(lc2_restricted.counts),
    )
    interp_err = np.interp(
        np.asarray(lc1_restricted.time),
        np.asarray(lc2_restricted.time),
        np.asarray(lc2_restricted.counts_err),
    )
    lc2_interp = Lightcurve(
        lc1_restricted.time,
        interp_counts,
        err=interp_err,
        gti=gti,
        err_dist=lc2_restricted.err_dist,
    )
    
    return lc1_restricted, lc2_interp


