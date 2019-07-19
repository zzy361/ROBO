import numpy as np
import pandas as pd


def mdd(timeseries):
    if len(timeseries) == 0:
        return 0
    else:
        timeseries = np.array(timeseries)
        return max(1 - timeseries / np.maximum.accumulate(timeseries))
