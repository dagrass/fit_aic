from lmfit import Model as _Model
from .._utils import _compute_aic_aicc


class Model(_Model):
    def fit(self, data, params=None, **kwargs):
        result = super().fit(data, params=params, **kwargs)
        aic, aicc = _compute_aic_aicc(result.chisqr, result.ndata, result.nvarys)
        result.aicc = aicc
        return result
