from typing import overload, Literal, Tuple, Callable, Any, Optional, Union
from scipy.optimize import curve_fit as _curve_fit
import numpy as np
from numpy.typing import NDArray
from .._utils import _compute_aic_aicc


@overload
def curve_fit(
    f: Callable,
    xdata: NDArray,
    ydata: NDArray,
    p0: Optional[NDArray] = None,
    sigma: Optional[Union[NDArray, float]] = None,
    absolute_sigma: bool = False,
    check_finite: Optional[bool] = None,
    bounds: Tuple[Union[float, NDArray], Union[float, NDArray]] = (-np.inf, np.inf),
    method: Optional[str] = None,
    jac: Optional[Callable] = None,
    *,
    full_output: Literal[False] = False,
    nan_policy: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[NDArray, NDArray]: ...


@overload
def curve_fit(
    f: Callable,
    xdata: NDArray,
    ydata: NDArray,
    p0: Optional[NDArray] = None,
    sigma: Optional[Union[NDArray, float]] = None,
    absolute_sigma: bool = False,
    check_finite: Optional[bool] = None,
    bounds: Tuple[Union[float, NDArray], Union[float, NDArray]] = (-np.inf, np.inf),
    method: Optional[str] = None,
    jac: Optional[Callable] = None,
    *,
    full_output: Literal[True],
    nan_policy: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[NDArray, NDArray, dict, str, int]: ...


def curve_fit(
    f: Callable,
    xdata: NDArray,
    ydata: NDArray,
    p0: Optional[NDArray] = None,
    sigma: Optional[Union[NDArray, float]] = None,
    absolute_sigma: bool = False,
    check_finite: Optional[bool] = None,
    bounds: Tuple[Union[float, NDArray], Union[float, NDArray]] = (-np.inf, np.inf),
    method: Optional[str] = None,
    jac: Optional[Callable] = None,
    *,
    full_output: bool = False,
    nan_policy: Optional[str] = None,
    **kwargs: Any,
) -> Union[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray, dict, str, int]]:
    """Fit a function to data using non-linear least squares.

    Wrapper around scipy.optimize.curve_fit that automatically computes AIC and AICc
    when full_output=True.

    Parameters
    ----------
    f : Callable
        The model function to fit.
    xdata : NDArray
        The independent variable data.
    ydata : NDArray
        The dependent variable data.
    p0 : NDArray, optional
        Initial parameter guess.
    sigma : NDArray or float, optional
        Standard deviation(s) for weighting.
    absolute_sigma : bool, default False
        If True, sigma is absolute values. If False, sigma is relative.
    check_finite : bool, optional
        Whether to check for non-finite data (NaN, inf).
    bounds : tuple of (NDArray, NDArray), default (-inf, inf)
        Lower and upper bounds for parameters.
    method : str, optional
        Optimization method to use.
    jac : Callable, optional
        Jacobian function.
    full_output : bool, default False
        If False, return (popt, pcov).
        If True, return (popt, pcov, infodict, mesg, ier) with AIC/AICc in infodict.
    nan_policy : str, optional
        How to handle NaN values.
    **kwargs
        Additional keyword arguments passed to scipy.optimize.curve_fit.

    Returns
    -------
    popt : NDArray
        Optimal parameter values.
    pcov : NDArray
        Covariance matrix of parameters.
    infodict : dict (only if full_output=True)
        Dictionary with optimization info, including 'aic' and 'aicc' keys.
    mesg : str (only if full_output=True)
        Optimization message.
    ier : int (only if full_output=True)
        Integer flag indicating success.
    """

    if not full_output:
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
