import numpy as np
from tsfresh.feature_extraction.feature_calculators import(
    agg_linear_trend
    )
from .rule import(
    TrendRule,
    VolatilityRule,
    ActiveLevelRule
)


class TimeSeriesAnalysis:
    def __init__(self) -> None:
        pass

    def __call__(self, ts):
        price = np.array(ts['close'])
        turnover = np.array(ts['turnover'])
        trend = TrendRule.to_text(_get_trend(price))
        volatility = VolatilityRule.to_text(_get_volatility(price))
        active_level = ActiveLevelRule.to_text(_get_active_level(turnover))
        return trend, volatility, active_level


def _get_trend(x):
    x = x.copy()
    x = x[-20:]
    if x.sum() < 1:
        return 2
    x0 = x[x > 0][0]
    x /= x0

    y = agg_linear_trend(
        x,
        param=[
            {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'},
            {'attr': 'pvalue', 'chunk_len': 5, 'f_agg': 'mean'}
        ])
    
    y = tuple(y)
    slope = y[0][1]
    pvalue = y[1][1]
    
    if pvalue > 0.1:
        state = 2
    elif slope > 0.01:
        state = 1
    elif slope < -0.01:
        state = 0
    else:
        state = 2
    
    return state


def _get_volatility(x):
    x = x.copy()
    x = np.log(x[1:]) - np.log(x[:-1])
    x = np.nan_to_num(x)
    x0 = x[:-5]
    s0 = (x0 - x0.mean()).std()
    x1 = x[-5:]
    s1 = (x1 - x1.mean()).std()
    return int(s1 > s0)


def _get_active_level(x):
    x = x.copy()
    q_high = np.quantile(x, q=0.7)
    q_low = np.quantile(x, q=0.3)
    x_mean = x[-5:].mean()
    if x_mean < q_low:
        state = 0
    elif x_mean > q_high:
        state = 1
    else:
        state = 2
    return state
