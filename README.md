# fit_aic
[![PyPI version](https://badge.fury.io/py/fit-aic.svg)](https://badge.fury.io/py/fit-aic)

AIC and AICc wrappers for scipy and lmfit, making it easy to compute Akaike information criteria for model comparison.

## Overview
`fit_aic` provides convenient wrappers around popular curve fitting libraries that automatically compute AIC (Akaike Information Criterion) and AICc (corrected AIC) values. This makes model comparison straightforward without manual calculations.

## Features
- **scipy.optimize.curve_fit wrapper**: Drop-in replacement with automatic AIC/AICc computation
- **lmfit.Model wrapper**: Drop-in replacement that adds AICc to the existing lmfit result
- **AIC and AICc calculations**: Properly formatted and corrected for sample size
- **Seamless integration**: Works with your existing scipy and lmfit code
- **Tested for accuracy**: Validated against lmfit results

## Background

The Akaike Information Criterion (AIC) and its corrected version for small sample sizes (AICc) is a metric for comparing statistical models that balances goodness of fit against model complexity [[1, 2, 3]](#references). A model with more parameters will always fit data better, but risks overfitting — AIC penalizes complexity to find the best tradeoff.

Given two models fit to the same data, the one with the **lower AIC is preferred**. The absolute value of AIC is not meaningful — only differences between models matter.

As an example we simulate data from a bi-exponential decay:

$$y = 3e^{-x/1} + 5e^{-x/10}$$

If we did not know the underlying analytical function of this process we might try fitting the data with different models to get an idea which model fits best. This is where the Akaike Information Criterion comes in. While increasing model complexity will reduce the residuals, there is no penalty for overfitting — AIC does provide this penalty. In our example we fit three models:

$$f_1(x) = a_1 e^{-x/t_1}$$

$$f_2(x) = a_1 e^{-x/t_1} + a_2 e^{-x/t_2}$$

$$f_3(x) = a_1 e^{-x/t_1} + a_2 e^{-x/t_2} + a_3 e^{-x/t_3}$$

The lowest AIC value can be used to select the best model that fits the data. This is illustrated in the figure below. Model 1 is plotted in blue, model 2 in orange, and model 3 in green. It appears model 2 and model 3 are almost identical, but model 2 is better as it has the lower AIC value — which makes sense given that the fits look the same but model 3 has 2 more free parameters.

![AIC comparison](other/model_comparison.png)

AICc is a corrected version of AIC for small sample sizes. As $n \to \infty$, AICc converges to AIC. It is recommended to always use AICc unless $n/k > 40$, where $n$ is the number of data points and $k$ the number of parameters.

A common rule of thumb for interpreting AIC differences ($\Delta AIC$):
- $\Delta AIC < 2$: models are essentially equivalent
- $2 < \Delta AIC < 10$: moderate evidence for the better model
- $\Delta AIC > 10$: strong evidence for the better model

## Installation
```bash
pip install fit_aic
```
Or with lmfit integration (optional):
```bash
pip install fit_aic[lmfit]
```

## Quick Start

### scipy wrapper
```python
import numpy as np
from fit_aic.scipy import curve_fit

# Define your model
def exponential_decay(x, A, tau):
    return A * np.exp(-x / tau)

# Fit with AIC/AICc computation
x = np.linspace(0, 10, 50)
y = exponential_decay(x, 3, 2) + np.random.normal(0, 0.1, size=x.shape)

popt, pcov, infodict, mesg, ier = curve_fit(
    exponential_decay, x, y,
    p0=[3, 2],
    full_output=True
)
print(f"AIC: {infodict['aic']:.2f}")
print(f"AICc: {infodict['aicc']:.2f}")
```

### lmfit wrapper
```python
import numpy as np
from fit_aic.lmfit import Model

def exponential_decay(x, A, tau):
    return A * np.exp(-x / tau)

x = np.linspace(0, 10, 50)
y = exponential_decay(x, 3, 2) + np.random.normal(0, 0.1, size=x.shape)

model = Model(exponential_decay)
result = model.fit(y, x=x, A=3, tau=2)
print(f"AIC: {result.aic:.2f}")
print(f"AICc: {result.aicc:.2f}")
```

### Model Comparison
```python
import numpy as np
from fit_aic.scipy import curve_fit

def model1(x, A1, tau1, A2, tau2):
    return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

def model2(x, A1, tau1):
    return A1 * np.exp(-x / tau1)

x = np.linspace(0, 20, 50)
y = model1(x, 3, 1, 5, 10) + np.random.normal(0, 0.25, size=x.shape)

result1 = curve_fit(model1, x, y, p0=[3, 1, 5, 10], full_output=True)
result2 = curve_fit(model2, x, y, p0=[3, 2], full_output=True)

aic1 = result1[2]['aic']
aic2 = result2[2]['aic']
print(f"Model 1 AIC: {aic1:.2f}")
print(f"Model 2 AIC: {aic2:.2f}")
print(f"Best model: {'Model 1' if aic1 < aic2 else 'Model 2'}")
```

## API Reference

### `fit_aic.scipy.curve_fit`
Wrapper around `scipy.optimize.curve_fit` with AIC/AICc support.

**Parameters:**
- All parameters are identical to `scipy.optimize.curve_fit`
- `full_output`: If `True`, returns 5-tuple with infodict containing `aic` and `aicc` (default: `False`)

**Returns:**
- If `full_output=False`: Returns `(popt, pcov)` — identical to scipy behavior
- If `full_output=True`: Returns `(popt, pcov, infodict, mesg, ier)` where `infodict` includes `aic` and `aicc` keys

### `fit_aic.lmfit.Model`
Subclass of `lmfit.Model` with AICc support. All existing lmfit behavior is preserved.

**Additional attributes on `ModelResult`:**
- `result.aicc`: Corrected AIC value

**Usage:**
```python
from fit_aic.lmfit import Model

model = Model(my_func)
result = model.fit(y, x=x, A=3, tau=2)
print(f"AIC: {result.aic:.2f}")   # lmfit built-in
print(f"AICc: {result.aicc:.2f}") # added by fit_aic
```

## Information Criteria

**AIC:**

$$AIC = n \ln(RSS/n) + 2k$$

**AICc:**

$$AICc = AIC + \frac{2k(k+1)}{n-k-1}$$

Where:
- $n$ = number of observations
- $k$ = number of parameters
- $RSS$ = residual sum of squares

AICc includes a correction for small sample sizes and converges to AIC as $n \to \infty$. It is recommended when $n/k < 40$.

## Development

Install with dev dependencies:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest tests/
```

## License
MIT

## References
<a id="references"></a>

[1] H. Akaike, "A New Look at the Statistical Model Identification," *IEEE Trans. Autom. Control* 19(6), 716–723 (1974). https://doi.org/10.1109/TAC.1974.1100705

[2] J. E. Cavanaugh, "Unifying the derivations for the Akaike and corrected Akaike information criteria," *Stat. & Probab. Lett.* 33(2), 201–208 (1997). https://doi.org/10.1016/S0167-7152(96)00128-9

[3] K. P. Burnham & D. R. Anderson, *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2nd ed.). Springer, New York (1998).
