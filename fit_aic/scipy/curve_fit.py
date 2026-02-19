from scipy.optimize import curve_fit as _curve_fit
import numpy as np
from .._utils import _compute_aic_aicc


def curve_fit(
    f,
    xdata,
    ydata,
    p0=None,
    sigma=None,
    absolute_sigma=False,
    check_finite=None,
    bounds=(-np.inf, np.inf),
    method=None,
    jac=None,
    *,
    full_output=False,
    nan_policy=None,
    **kwargs,
):
    """Fit a function to data using non-linear least squares."""

    if full_output == False:
        return _curve_fit(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            method=method,
            jac=jac,
            full_output=False,
            nan_policy=nan_policy,
            **kwargs,
        )
    else:
        popt, pcov, infodict, mesg, ier = _curve_fit(
            f,
            xdata,
            ydata,
            p0=p0,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            check_finite=check_finite,
            bounds=bounds,
            method=method,
            jac=jac,
            full_output=True,
            nan_policy=nan_policy,
            **kwargs,
        )
        residuals = infodict["fvec"]
        rss = np.sum(residuals**2)
        n = len(ydata)
        k = len(popt)
        aic, aicc = _compute_aic_aicc(rss, n, k)
        infodict.update({"aic": aic, "aicc": aicc})
        return popt, pcov, infodict, mesg, ier
