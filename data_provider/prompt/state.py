import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td


def get_trend_and_volatility_state(price_series, window_size, high_q=0.7, low_q=0.3, volatility_q=0.7):
    # price_series (seq_len, N)

    change_rate = np.log(price_series[1:]) - np.log(price_series[:-1])
    # change_rate (seq_len-1, N)

    trend_states, volatility_states, mask = [], [], []
    for w in range(window_size, change_rate.shape[0]+1):
        window = change_rate[w-window_size: w]
        window_mask = ~np.isnan(window[-1])
        window = np.nan_to_num(window)
        mask.append(window_mask)

        window_mean = np.mean(window, axis=0)
        quantiles_high = np.quantile(window[:,window_mask], high_q)
        quantiles_low = np.quantile(window[:,window_mask], low_q)
        window_state = 2 * np.ones(window_mean.shape, dtype=np.int8)
        window_state[window_mean >= quantiles_high] = 1
        window_state[window_mean <= quantiles_low] = 0
        trend_states.append(window_state)

        window_volatility = np.var(window, axis=0)
        volatility_high = np.quantile(window_volatility[window_mask], volatility_q)
        volatility_state = np.zeros(window_volatility.shape, dtype=np.int8)
        volatility_state[window_volatility >= volatility_high] = 1
        volatility_states.append(volatility_state)

    trend_states = np.vstack(trend_states)
    volatility_states = np.vstack(volatility_states)
    mask = np.vstack(mask)
    
    return trend_states, volatility_states, mask


def get_turnover_state(turnover_series, window_size, high_q=0.7, low_q=0.3):
    # volume_series (seq_len, N)

    turnover_series = turnover_series[1:]

    volume_states, mask = [], []
    for w in range(window_size, turnover_series.shape[0]+1):
        window = turnover_series[w-window_size: w]
        window_mask = ~np.isnan(window[-1])
        window = np.nan_to_num(window)
        mask.append(window_mask)

        window_mean = np.mean(window, axis=0)
        quantiles_high = np.quantile(window[:,window_mask], high_q)
        quantiles_low = np.quantile(window[:,window_mask], low_q)
        window_state = 2 * np.ones(window_mean.shape, dtype=np.int8)
        window_state[window_mean >= quantiles_high] = 1
        window_state[window_mean <= quantiles_low] = 0
        volume_states.append(window_state)
    
    volume_states = np.vstack(volume_states)

    return volume_states


def get_date_states(dates, delivery_dates, holiday_dates, time_delta=3):
    dates = pd.DataFrame({'date': pd.to_datetime(dates)})

    delivery_states = np.zeros(len(dates), dtype=np.int8)
    holiday_states = np.zeros(len(dates), dtype=np.int8)

    dinc = td(days=time_delta)

    for d in delivery_dates:
        d = dt.strptime(d, '%Y-%m-%d')
        start_d = d - dinc
        end_d = d + dinc
        in_range = (dates['date'] >= start_d) & (dates['date'] <= end_d)
        delivery_states[in_range] = 1
    
    for d in holiday_dates:
        d = dt.strptime(d, '%Y-%m-%d')
        start_d = d - dinc
        end_d = d + dinc
        in_range = (dates['date'] >= start_d) & (dates['date'] <= end_d)
        holiday_states[in_range] = 1
    
    return delivery_states, holiday_states
