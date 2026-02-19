from scipy.optimize import curve_fit as _curve_fit
from fit_aic.scipy import curve_fit
from lmfit import Model
import numpy as np
import pytest


def model(x, A1, tau1, A2, tau2):
    return A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)


@pytest.fixture
def simulate_data():
    np.random.seed(0)
    x = np.linspace(0, 20, 30)
    y = model(x, 3, 1, 5, 10) + np.random.normal(0, 0.25, size=x.shape)

    return x, y


@pytest.fixture
def simulate_data_large():
    np.random.seed(0)
    x = np.linspace(0, 20, 1000)
    y = model(x, 3, 1, 5, 10) + np.random.normal(0, 0.25, size=x.shape)

    return x, y


def test_curve_fit_standard_output(simulate_data):
    x, y = simulate_data
    popt_scipy, pcov_scipy = _curve_fit(model, x, y, p0=[3, 1, 5, 3], full_output=False)
    popt_ours, pcov_ours = curve_fit(model, x, y, p0=[3, 1, 5, 3], full_output=False)
    np.testing.assert_allclose(popt_scipy, popt_ours)
    np.testing.assert_allclose(pcov_scipy, pcov_ours)


def test_curve_fit_full_output(simulate_data):
    x, y = simulate_data
    output = curve_fit(model, x, y, p0=[3, 1, 5, 3], full_output=True)
    assert len(output) == 5
    popt, pcov, infodict, mesg, ier = output
    assert isinstance(popt, np.ndarray)
    assert isinstance(pcov, np.ndarray)
    assert isinstance(infodict, dict)
    assert isinstance(mesg, str)
    assert isinstance(ier, int)
    assert "aic" in infodict
    assert "aicc" in infodict
    assert isinstance(infodict["aic"], float)
    assert isinstance(infodict["aicc"], float)


def test_curve_fit_aic_comparison(simulate_data):
    x, y = simulate_data
    popt_ours, pcov_ours, infodict_ours, mesg_ours, ier_ours = curve_fit(
        model,
        x,
        y,
        p0=[3, 1, 5, 3],
        full_output=True,
    )
    model_lmfit = Model(model)
    result_lmfit = model_lmfit.fit(y, x=x, A1=3, tau1=1, A2=5, tau2=3)

    assert np.isclose(infodict_ours["aic"], result_lmfit.aic, atol=1e-2)


def test_curve_fit_aicc_comparison(simulate_data_large):
    x, y = simulate_data_large
    popt_ours, pcov_ours, infodict_ours, mesg_ours, ier_ours = curve_fit(
        model,
        x,
        y,
        p0=[3, 1, 5, 3],
        full_output=True,
    )
    print(f"AIC: {infodict_ours['aic']:.2f}, AICc: {infodict_ours['aicc']:.2f}")
    assert np.isclose(
        infodict_ours["aicc"],
        infodict_ours["aic"],
        atol=1e-1,
    )
