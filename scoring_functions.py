# -------------------------------------------------------#
# BN scoring functions
# -------------------------------------------------------#
from data_type import ll_metric
import numpy as np


# Hill climbing algorithm for ets
# data: data_type data
# xi: Node xi
# pa_xi: Parents of xi
# metric: scoring function (log_likelihood 'll', BIC 'bic', or AIC 'aic', or custom 'custom')
# metric_params: Parameters for the scoring function
# ll_in: log-likelihood (optional)


def score_function(data, xi, pa_xi, metric='bic', metric_params=None, ll_in = None, weights = None):
    
    if ll_in is None:
        ll = ll_metric(xi, pa_xi, data)
    else:
        ll = ll_in
    if weights is None:
        nrows = data.nrows
    else:
        nrows = sum(weights)
        
        
    if metric.lower() == 'll':
        return ll
    elif metric.lower() == 'aic':
        qi = np.prod([len(data.classes[p]) for p in pa_xi])
        ri = len(data.classes[xi])
        b = (ri - 1) * qi
        return ll - b
    elif metric.lower() == 'bic':
        qi = np.prod([len(data.classes[p]) for p in pa_xi])
        ri = len(data.classes[xi])
        b = (ri - 1) * qi
        return ll - (1.0 / 2) * np.log(nrows) * b
    elif metric.lower() == 'custom':
        qi = np.prod([len(data.classes[p]) for p in pa_xi])
        ri = len(data.classes[xi])
        b = (ri - 1) * qi
        if metric_params is None:
            return ll - b
        else:
            return ll - b * metric_params[0]
    else:
        str = 'Scoring function ', metric, ' is not supported.'
        raise ValueError(str)

