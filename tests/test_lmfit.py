from fit_aic.lmfit import Model
from lmfit import Model as _Model
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


def test_model_aicc(simulate_data):
    x, y = simulate_data
    model_lmfit = Model(model)
    result_lmfit = model_lmfit.fit(y, x=x, A1=3, tau1=1, A2=5, tau2=3)

    assert hasattr(result_lmfit, "aicc")
    assert isinstance(result_lmfit.aicc, float)


def test_model_aicc_limit(simulate_data_large):
    x, y = simulate_data_large
    model_lmfit = Model(model)
    result_lmfit = model_lmfit.fit(y, x=x, A1=3, tau1=1, A2=5, tau2=3)
    assert np.isclose(result_lmfit.aicc, result_lmfit.aic, atol=1e-1)
