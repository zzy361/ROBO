import numpy as np
from datetime import datetime
import pandas as pd


def daily_return_ratio(asset, start, end):
    asset_array = np.array(asset.iloc[start:end].as_matrix(columns=None))

    return_ratio = []
    for i in range(0, len(asset.index[start:end]) - 1):
        return_ratio.append(asset_array[i + 1] / asset_array[i] - 1)
    result = pd.DataFrame(return_ratio, index=asset.index[start:end].delete(0), columns=None)
    return result


def standard_deviation(asset, start, end):
    daily_return = daily_return_ratio(asset, start, end)
    ratio = np.array(daily_return.as_matrix(columns=None)).T
    result = np.sqrt(ratio.std())
    return result


def total_return_ratio(asset, start, end):
    daily_return = daily_return_ratio(asset, start, end)
    ratio = np.array(daily_return.as_matrix(columns=None)) + 1
    result = np.cumprod(ratio)[-1] - 1
    return result


def annualized_return_ratio(asset, start, end):

    real_return = total_return_ratio(asset, start, end)
    result = (real_return + 1) ** (250 / len(asset)) - 1
    return result


def max_draw_down(asset, start, end):
    result = []
    if end - start > 3:
        daily_return = daily_return_ratio(asset, start, end)
        array_temp = np.array(daily_return.as_matrix(columns=None)) + 1

        result.append(max(1 - np.cumprod(array_temp) / np.maximum.accumulate(np.cumprod(array_temp))))
        c = list(1 - np.cumprod(array_temp) / np.maximum.accumulate(np.cumprod(array_temp)))
        result.append(start + c.index(max(c)))
    else:
        result.append(0)
        result.append(0)
    return result

def sharpe_ratio(asset, start, end, risk_free = 0):
    return (annualized_return_ratio(asset, start, end) - risk_free)/ standard_deviation(asset, start, end)
