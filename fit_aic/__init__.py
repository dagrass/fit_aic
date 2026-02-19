"""fit_aic: AIC and AICc wrappers for scipy and lmfit

This package provides convenient wrappers around popular curve fitting libraries
that automatically compute AIC (Akaike Information Criterion) and AICc (corrected AIC)
values for model comparison.

Submodules
----------
scipy : scipy.optimize.curve_fit wrapper with AIC/AICc support
lmfit : lmfit.Model wrapper that adds AICc to fit results

Examples
--------
>>> from fit_aic.scipy import curve_fit
>>> import numpy as np
>>>
>>> def model(x, a, b):
...     return a * np.exp(-b * x)
>>>
>>> x = np.linspace(0, 10, 50)
>>> y = model(x, 3, 0.5) + np.random.normal(0, 0.1, 50)
>>>
>>> popt, pcov, infodict, mesg, ier = curve_fit(
...     model, x, y, p0=[3, 0.5], full_output=True
... )
>>> print(f"AIC: {infodict['aic']:.2f}, AICc: {infodict['aicc']:.2f}")
"""

__all__ = []
