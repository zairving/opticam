from typing import Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from opticam.fitting.models import power_law, straight_line

def fit_rms_vs_flux(
    data: Dict,
    ) -> Dict[str, Dict[str, NDArray]]:
    """
    Iteratively fit a straight line (in log space) to the RMS vs flux plots for each catalog. This can be used to
    identify variable sources and good comparison sources.
    
    Parameters
    ----------
    data : Dict
        The RMS vs flux data.
    
    Returns
    -------
    Dict[str, Dict[str, NDArray]]
        The power law fits for each filter `{filter: {'flux': NDArray, 'rms': NDArray}}`.
    """
    
    pl_fits = {}
    
    for fltr in data.keys():
        rms, flux = [], []
        for source_number, values in data[fltr].items():
            rms.append(values['rms'])
            flux.append(values['flux'])
        
        x = np.array(flux)
        y = np.asarray(rms)
        
        try:
            converged = False
            prev, prev_err = None, None
            while not converged:
                log_x = np.log10(flux)
                log_y = np.log10(rms)
                
                popt, pcov = curve_fit(
                    straight_line,
                    log_x,
                    log_y,
                    )
                perr = np.sqrt(np.diag(pcov))
                
                if prev is not None and prev_err is not None:
                    converged = np.allclose(popt, prev, atol=np.sqrt(perr**2 + prev_err**2))
                
                # remove largest outliers
                model = straight_line(log_x, *popt)
                r = log_y - model
                i = np.argmax(r)
                
                rms.pop(i)
                flux.pop(i)
                
                prev = popt
                prev_err = perr
        except:
            popt, pcov = curve_fit(
                    straight_line,
                    np.log10(x),
                    np.log10(y),
                    )
        
        y_model = power_law(
            x,
            10**(popt[1]),
            popt[0],
            )
        
        pl_fits[fltr] = {
            'flux': x,
            'rms': y_model,
        }
    
    return pl_fits
