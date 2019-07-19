
'''
@author: weixiang
@file: IndicatTest.py
@time: 2017/11/24 11:09
@desc:

回测指标计算，包括计算连续上涨天数，连续下跌天数，最大回撤，夏普比，年化收益，量化波动

'''


from __future__ import division

import pandas as pd
import numpy as np
from scipy import stats
from six import iteritems
import copy


from data.DataProcess.Periodic_selection import *
from .utils import nanmean, nanstd, nanmin, up, down, roll
from .periods import ANNUALIZATION_FACTORS, APPROX_BDAYS_PER_YEAR
from .periods import DAILY, WEEKLY, MONTHLY, YEARLY


def _adjust_returns(returns, adjustment_factor):
    """
    Returns the returns series adjusted by adjustment_factor. Optimizes for the
    case of adjustment_factor being 0 by returning returns itself, not a copy!

    Parameters
    ----------
    returns : pd.Series or np.ndarray
    adjustment_factor : pd.Series or np.ndarray or float or int

    Returns
    -------
    pd.Series or np.ndarray
    """
    if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
        return returns
    return returns - adjustment_factor


def annualization_factor(period, annualization):
    """
    Return annualization factor from period entered or if a custom
    value is passed in.

    Parameters
    ----------
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Annualization factor.
    """
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def cum_returns(returns, starting_value=0):
    """
    计算对数收益序列的累积收益

    Parameters
    ----------
    returns : pd.Series, np.ndarray, or pd.DataFrame
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902.
        - Also accepts two dimensional data. In this case,
            each column is cumulated.
    starting_value : float, optional，0表示返回收益率序列，可看作是初始资金
       The starting returns.

    Returns
    -------
    pd.Series, np.ndarray, or pd.DataFrame
        Series of cumulative returns.

    Notes
    -----
    For increased numerical accuracy, convert input to log returns
    where it is possible to sum instead of multiplying.
    PI((1+r_i)) - 1 = exp(ln(PI(1+r_i)))
                    = exp(SIGMA(ln(1+r_i))
    """


    if len(returns) < 1:
        return type(returns)([])

    if np.any(np.isnan(returns)):
        returns = returns.copy()
        returns[np.isnan(returns)] = 0.

    df_cum = (returns + 1).cumprod(axis=0)

    if starting_value == 0:
        return df_cum - 1
    else:
        return df_cum * starting_value


def GetReturnData(result):
    """
    输入原始数据，返回原始数据的简单收益矩阵，去掉第一行

    :param result:
    :return:
    """
    temrawData = result
    items = list(temrawData.columns)
    returndata = pd.DataFrame()

    for it in items:
        returnname=it+"_return"
        temrawData[returnname] = 100*temrawData[it].diff()/temrawData[it]
        returndata[it] = temrawData[returnname]
    returndata = returndata.dropna()

    return returndata



def cum_returns_final(returns, starting_value=0):
    """
    计算简单收益序列的累积收益,返回最后的值

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy as a percentage, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902.
    starting_value : float, optional
       The starting returns.

    Returns
    -------
    float

    """

    if len(returns) == 0:
        return np.nan

    return cum_returns(np.asanyarray(returns),
                       starting_value=starting_value)[-1]


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by week, month, or year.

    Parameters
    ----------
    returns : pd.Series
       Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    Returns
    -------
    pd.Series
        Aggregated returns.
    """

    def cumulate_returns(x):
        return cum_returns(x).iloc[-1]

    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )

    return returns.groupby(grouping).apply(cumulate_returns)


def max_drawdown(returns):
    """
    计算最大回撤

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.

    Returns
    -------
    float
        Maximum drawdown.

    Note
    -----
    See https://en.wikipedia.org/wiki/Drawdown_(economics) for more details.
    """

    if len(returns) < 1:
        return np.nan

    if type(returns) == pd.Series:
        returns = returns.values

    cumulative = np.insert(cum_returns(returns, starting_value=100), 0, 100)
    max_return = np.fmax.accumulate(cumulative)

    return nanmin((cumulative - max_return) / max_return)


def annual_return(returns, period=DAILY, annualization=None):
    """年平均收益率

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Annual Return as CAGR (Compounded Annual Growth Rate).

    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    num_years = float(len(returns)) / ann_factor
    start_value = 100

    end_value = cum_returns(np.asanyarray(returns),
                            starting_value=start_value)[-1]
    cum_returns_final = (end_value - start_value) / start_value
    annual_return = (1. + cum_returns_final) ** (1. / num_years) - 1

    return annual_return


def annual_volatility(returns, period=DAILY, alpha=2.0,
                      annualization=None):
    """
    计算年化波动率

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Periodic returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    alpha : float, optional
        Scaling relation (Levy stability exponent).
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Annual volatility.
    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    volatility = nanstd(returns, ddof=1) * (ann_factor ** (1.0 / alpha))

    return volatility


def calmar_ratio(returns, period=DAILY, annualization=None):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.


    Returns
    -------
    float
        Calmar ratio (drawdown ratio) as float. Returns np.nan if there is no
        calmar ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Calmar_ratio for more details.
    """

    max_dd = max_drawdown(returns=returns)
    if max_dd < 0:
        temp = annual_return(
            returns=returns,
            period=period,
            annualization=annualization
        ) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def omega_ratio(returns, risk_free=0.0, required_return=0.0,
                annualization=APPROX_BDAYS_PER_YEAR):
    """Determines the Omega ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period
    required_return : float, optional
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g. An annual minimum
        acceptable return of 100 will translate to a minimum acceptable
        return of 0.018.
    annualization : int, optional
        Factor used to convert the required_return into a daily
        value. Enter 1 if no time period conversion is necessary.

    Returns
    -------
    float
        Omega ratio.

    Note
    -----
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.

    """

    if len(returns) < 2:
        return np.nan

    if annualization == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** \
            (1. / annualization) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sharpe_ratio(returns, risk_free=0, period=DAILY, annualization=None):
    """
    Determines the Sharpe ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    risk_free : int, float
        Constant risk-free return throughout the period.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Sharpe ratio.

        np.nan
            If insufficient length of returns or if if adjusted returns are 0.

    Note
    -----
    See https://en.wikipedia.org/wiki/Sharpe_ratio for more details.

    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    returns_risk_adj = np.asanyarray(_adjust_returns(returns, risk_free))
    returns_risk_adj = returns_risk_adj[~np.isnan(returns_risk_adj)]

    if nanstd(returns_risk_adj, ddof=1) == 0:
        return np.nan

    return nanmean(returns_risk_adj) / nanstd(returns_risk_adj, ddof=1) * \
        np.sqrt(ann_factor)


def sortino_ratio(returns, required_return=0, period=DAILY,
                  annualization=None, _downside_risk=None):
    """
    Determines the Sortino ratio of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
    _downside_risk : float, optional
        The downside risk of the given inputs, if known. Will be calculated if
        not provided.

    Returns
    -------
    float, pd.Series

        depends on input type
        series ==> float
        DataFrame ==> pd.Series

        Annualized Sortino ratio.

    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    adj_returns = _adjust_returns(returns, required_return)
    mu = nanmean(adj_returns, axis=0)
    dsr = (_downside_risk if _downside_risk is not None
           else downside_risk(returns, required_return,
                              period=period, annualization=annualization))
    sortino = mu / dsr
    return sortino * ann_factor


def downside_risk(returns, required_return=0, period=DAILY,
                  annualization=None):
    """
    Determines the downside deviation below a threshold

    Parameters
    ----------
    returns : pd.Series or np.ndarray or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    required_return: float / series
        minimum acceptable return
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float, pd.Series
        depends on input type
        series ==> float
        DataFrame ==> pd.Series

        Annualized downside deviation

    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    downside_diff = _adjust_returns(returns, required_return).copy()
    mask = downside_diff > 0
    downside_diff[mask] = 0.0
    squares = np.square(downside_diff)
    mean_squares = nanmean(squares, axis=0)
    dside_risk = np.sqrt(mean_squares) * np.sqrt(ann_factor)

    if len(returns.shape) == 2 and isinstance(returns, pd.DataFrame):
        dside_risk = pd.Series(dside_risk, index=returns.columns)
    return dside_risk


def excess_sharpe(returns, factor_returns):
    """
    Determines the Excess Sharpe of a strategy.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    factor_returns: float / series
        Benchmark return to compare returns against.

    Returns
    -------
    float
        The excess sharpe.

    Note
    -----
    The excess Sharpe is a simplified Information Ratio that uses
    tracking error rather than "active risk" as the denominator.

    """
    if len(returns) < 2:
        return np.nan

    active_return = _adjust_returns(returns, factor_returns)
    tracking_error = nanstd(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return nanmean(active_return) / tracking_error


def _aligned_series(*many_series):
    """
    Return a new list of series containing the data in the input series, but
    with their indices aligned. NaNs will be filled in for missing values.

    Parameters
    ----------
    many_series : list[pd.Series]

    Returns
    -------
    aligned_series : list[pd.Series]

        A new list of series containing the data in the input series, but
        with their indices aligned. NaNs will be filled in for missing values.

    """
    return [series
            for col, series in iteritems(pd.concat(many_series, axis=1))]


def alpha_beta(returns, factor_returns, risk_free=0.0, period=DAILY,
               annualization=None):
    """Calculates annualized alpha and beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Alpha.
    float
        Beta.

    """
    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan, np.nan

    return alpha_beta_aligned(*_aligned_series(returns, factor_returns),
                              risk_free=risk_free, period=period,
                              annualization=annualization)


def alpha_beta_aligned(returns, factor_returns, risk_free=0.0, period=DAILY,
                       annualization=None):
    """Calculates annualized alpha and beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    Returns
    -------
    float
        Alpha.
    float
        Beta.
    """
    b = beta_aligned(returns, factor_returns, risk_free)
    a = alpha_aligned(returns, factor_returns, risk_free, period,
                      annualization, _beta=b)
    return a, b


def alpha(returns, factor_returns, risk_free=0.0, period=DAILY,
          annualization=None, _beta=None):
    """Calculates annualized alpha.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~financialMetrics.stats.annual_return`.
    _beta : float, optional
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.

    Returns
    -------
    float
        Alpha.
    """
    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan

    return alpha_aligned(*_aligned_series(returns, factor_returns),
                         risk_free=risk_free, period=period,
                         annualization=annualization, _beta=_beta)


def alpha_aligned(returns, factor_returns, risk_free=0.0, period=DAILY,
                  annualization=None, _beta=None):
    """Calculates annualized alpha.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~financialMetrics.stats.annual_return`.
    _beta : float, optional
        The beta for the given inputs, if already known. Will be calculated
        internally if not provided.

    Returns
    -------
    float
        Alpha.
    """
    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    if _beta is None:
        _beta = beta_aligned(returns, factor_returns, risk_free)

    adj_returns = _adjust_returns(returns, risk_free)
    adj_factor_returns = _adjust_returns(factor_returns, risk_free)
    alpha_series = adj_returns - (_beta * adj_factor_returns)

    return nanmean(alpha_series) * ann_factor


def beta(returns, factor_returns, risk_free=0.0):
    """Calculates beta.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    factor_returns : pd.Series
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    Returns
    -------
    float
        Beta.
    """
    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan

    return beta_aligned(*_aligned_series(returns, factor_returns),
                        risk_free=risk_free)


def beta_aligned(returns, factor_returns, risk_free=0.0):
    """Calculates beta.

    If they are pd.Series, expects returns and factor_returns have already
    been aligned on their labels.  If np.ndarray, these arguments should have
    the same shape.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
         Daily noncumulative returns of the factor to which beta is
         computed. Usually a benchmark such as the market.
         - This is in the same style as returns.
    risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    Returns
    -------
    float
        Beta.
    """

    if len(returns) < 2 or len(factor_returns) < 2:
        return np.nan

    joint = np.vstack([_adjust_returns(returns, risk_free),
                       factor_returns])
    joint = joint[:, ~np.isnan(joint).any(axis=0)]
    if joint.shape[1] < 2:
        return np.nan

    cov = np.cov(joint, ddof=0)

    if np.absolute(cov[1, 1]) < 1.0e-30:
        return np.nan

    return cov[0, 1] / cov[1, 1]


def stability_of_timeseries(returns):
    """Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.

    Returns
    -------
    float
        R-squared.

    """
    if len(returns) < 2:
        return np.nan

    returns = np.asanyarray(returns)
    returns = returns[~np.isnan(returns)]

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)),
                            cum_log_returns)[2]

    return rhat ** 2


def tail_ratio(returns):
    """Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
         - See full explanation in :func:`~financialMetrics.stats.cum_returns`.

    Returns
    -------
    float
        tail ratio

    """

    if len(returns) < 1:
        return np.nan

    returns = np.asanyarray(returns)

    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))


def cagr(returns, period=DAILY, annualization=None):
    """
    Compute compound annual growth rate.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~financialMetrics.stats.annual_return`.

    Returns
    -------
    float, np.nan
        The CAGR value.

    """
    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)
    no_years = len(returns) / float(ann_factor)

    ending_value = cum_returns(np.asanyarray(returns), starting_value=1)[-1]

    return ending_value ** (1. / no_years) - 1


def capture(returns, factor_returns, period=DAILY):
    """
    Compute capture ratio.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Returns of the strategy, noncumulative.
        - See full explanation in :func:`~financialMetrics.stats.cum_returns`.
    factor_returns : pd.Series or np.ndarray
        Noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    Returns
    -------
    float, np.nan
        The capture ratio.

    Notes
    -----
    See http://www.investopedia.com/terms/u/up-market-capture-ratio.asp for
    details.
    """
    return (annual_return(returns, period=period) /
            annual_return(factor_returns, period=period))


def up_capture(returns, factor_returns, **kwargs):
    """
    Compute the capture ratio for periods when the benchmark return is positive

    Parameters
    ----------
    see documentation for `capture`.

    Returns
    -------
    float, np.nan

    Notes
    -----
    See http://www.investopedia.com/terms/u/up-market-capture-ratio.asp for
    more information.
    """
    return up(returns, factor_returns, function=capture, **kwargs)


def down_capture(returns, factor_returns, **kwargs):
    """
    Compute the capture ratio for periods when the benchmark return is negative

    Parameters
    ----------
    see documentation for `capture`.

    Returns
    -------
    float, np.nan

    Note
    ----
    See http://www.investopedia.com/terms/d/down-market-capture-ratio.asp for
    more information.
    """
    return down(returns, factor_returns, function=capture, **kwargs)


def up_down_capture(returns, factor_returns, **kwargs):
    """
    Computes the ratio of up_capture to down_capture.

    Parameters
    ----------
    see documentation for `capture`.

    Returns
    -------
    float
        the updown capture ratio
    """
    return (up_capture(returns, factor_returns, **kwargs) /
            down_capture(returns, factor_returns, **kwargs))


def up_alpha_beta(returns, factor_returns, **kwargs):
    """
    Computes alpha and beta for periods when the benchmark return is positive.

    Parameters
    ----------
    see documentation for `alpha_beta`.

    Returns
    -------
    float
        Alpha.
    float
        Beta.
    """
    return up(returns, factor_returns, function=alpha_beta_aligned, **kwargs)


def down_alpha_beta(returns, factor_returns, **kwargs):
    """
    Computes alpha and beta for periods when the benchmark return is negative.

    Parameters
    ----------
    see documentation for `alpha_beta`.

    Returns
    -------
    float
        Alpha.
    float
        Beta.
    """
    return down(returns, factor_returns, function=alpha_beta_aligned, **kwargs)


def roll_up_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the up capture measure over a rolling window.

    Parameters
    ----------
    see documentation for `capture` (pass all args, kwargs required)

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, factor_returns, window=window, function=up_capture,
                **kwargs)


def roll_down_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the down capture measure over a rolling window.

    Parameters
    ----------
    see documentation for `capture` (pass all args, kwargs required)

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, factor_returns, window=window, function=down_capture,
                **kwargs)


def roll_up_down_capture(returns, factor_returns, window=10, **kwargs):
    """
    Computes the up/down capture measure over a rolling window.

    Parameters
    ----------
    see documentation for `capture` (pass all args, kwargs required)

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, factor_returns, window=window,
                function=up_down_capture, **kwargs)


def roll_max_drawdown(returns, window=10, **kwargs):
    """
    Computes the max_drawdown measure over a rolling window.

    Parameters
    ----------
    see documentation for `max_drawdown` (pass all args, kwargs required)

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, window=window, function=max_drawdown, **kwargs)


def roll_alpha_beta(returns, factor_returns, window=10, **kwargs):
    """
    Computes the alpha_beta measure over a rolling window.

    Parameters
    ----------
    see documentation for `alpha_beta` (pass all args, kwargs required)

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, factor_returns, window=window,
                function=alpha_beta_aligned, **kwargs)


def roll_sharpe_ratio(returns, window=10, **kwargs):
    """
    Computes the sharpe ratio measure over a rolling window.

    Parameters
    ----------
    see documentation for `sharpe_ratio` (pass all args, kwargs required)

    window : int, required
        Size of the rolling window in terms of the periodicity of the data.
        - eg window = 60, periodicity=DAILY, represents a rolling 60 day window
    """
    return roll(returns, window=window, function=sharpe_ratio, **kwargs)


def value_at_risk(returns, cutoff=0.05):
    """
    Value at risk (VaR) of a returns stream.

    Parameters
    ----------
    returns : pandas.Series or 1-D numpy.array
        Non-cumulative daily returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.05.

    Returns
    -------
    VaR : float
        The VaR value.
    """
    return np.percentile(returns, 100 * cutoff)


def conditional_value_at_risk(returns, cutoff=0.05):
    """
    Conditional value at risk (CVaR) of a returns stream.

    CVaR measures the expected single-day returns of an asset on that asset's
    worst performing days, where "worst-performing" is defined as falling below
    ``cutoff`` as a percentile of all daily returns.

    Parameters
    ----------
    returns : pandas.Series or 1-D numpy.array
        Non-cumulative daily returns.
    cutoff : float, optional
        Decimal representing the percentage cutoff for the bottom percentile of
        returns. Defaults to 0.05.

    Returns
    -------
    CVaR : float
        The CVaR value.
    """

    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])

def wxTest():
    print ("this is a  test")


def getFrequencyData(rawdata,fre="day"):
    """

    :param rawdata:     index 为时间datetime格式的pd
    :param fre:         str="day";"week","month" 要获取的周期

    :return:             pd 新的index
    """

    datalist = list(rawdata.index)
    items = list(rawdata.columns)

    result = pd.DataFrame()
    for it in items:

        nav_list = list(rawdata[it])
        df = freq_df_construct(datalist, nav_list, fre)
        result[it] = df["nav"]
    return   result




def daily_return_ratio(asset, start, end):
    """

    :param asset:
    :param start:
    :param end:
    :return:
    """
    asset_array = np.array(asset.iloc[start:end].as_matrix(columns=None))

    return_ratio = []
    for i in range(0, len(asset.index[start:end]) - 1):
        return_ratio.append(asset_array[i + 1] / asset_array[i] - 1)
    result = pd.DataFrame(return_ratio, index=asset.index[start:end].delete(0), columns=None)
    return result


def standard_deviation(asset, start, end):
    daily_return = daily_return_ratio(asset, start, end)
    ratio = np.array(daily_return.as_matrix(columns=None)).T
    result = ratio.std()
    return result


def find_friday(start_day, end_day):
    friday_list = []
    i = start_day
    while i <= end_day:
        if i.isocalendar()[2] == 5:
            friday_list.append(i)
        i += timedelta(days=1)
    return friday_list


def find_month_end_day(start_day, end_day):
    month_end_day_list = []
    i = start_day
    while i <= end_day:
        if i.day == calendar.monthrange(i.year, i.month)[1]:
            month_end_day_list.append(i)
        i += timedelta(days=1)
    return month_end_day_list


def freq_df_construct(date_list, nav_list, frequency):
    """

    :param date_list:
                        list

    :param nav_list:

    :param frequency:
                        week
    :return:
    """
    nav_df = pd.DataFrame(nav_list, index=date_list, columns=['nav'])
    result = pd.DataFrame(columns=['date_label', 'nav'])
    choosen_index = []

    if frequency == 'day':
        date_label_list = [str(i.year) + 'd' + str(i.timetuple().tm_yday) for i in nav_df.index]
        result['date_label'] = date_label_list
        result['nav'] = nav_list
        result.index = date_list
    elif frequency == 'week':
        friday_list = find_friday(start_day=date_list[0], end_day=date_list[-1])
        for i in friday_list:
            choosen_index.append(nav_df.index[nav_df.index <= i].max())
        choosen_index1 = list(set(choosen_index))
        choosen_index1.sort(key=choosen_index.index)
        date_label_list = [str(i.year) + 'w' + str(i.isocalendar()[1]) for i in choosen_index1]
        new_nav_df = nav_df.loc[choosen_index1, :]
        result['date_label'] = date_label_list
        result['nav'] = list(new_nav_df['nav'])
        result.index = new_nav_df.index
    elif frequency == 'month':
        month_end_day_list = find_month_end_day(start_day=nav_df.index[0], end_day=nav_df.index[-1])
        for i in month_end_day_list:
            choosen_index.append(nav_df.index[nav_df.index <= i].max())
        choosen_index1 = list(set(choosen_index))
        choosen_index1.sort(key=choosen_index.index)
        date_label_list = [str(i.year) + 'm' + str(i.month) for i in choosen_index1]
        new_nav_df = nav_df.loc[choosen_index1, :]
        result['date_label'] = date_label_list
        result['nav'] = list(new_nav_df['nav'])
        result.index = new_nav_df.index
    return result


def volatility_cal(date_list, nav_list, frequency):
    nav_df = freq_df_construct(date_list, nav_list, frequency)
    new_df = nav_df
    del new_df['date_label']
    if frequency == 'day':
        return standard_deviation(new_df, 0, len(new_df)) * np.sqrt(252)
    elif frequency == 'week':
        return standard_deviation(new_df, 0, len(new_df)) * np.sqrt(52)
    if frequency == 'month':
        return standard_deviation(new_df, 0, len(new_df)) * np.sqrt(12)



SIMPLE_STAT_FUNCS = [
    cum_returns_final,
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    stats.skew,
    stats.kurtosis,
    tail_ratio,
    cagr,
    value_at_risk,
    conditional_value_at_risk,




    wxTest,
    GetReturnData,
    getFrequencyData,

]

FACTOR_STAT_FUNCS = [
    excess_sharpe,
    alpha,
    beta,
    capture,
    up_capture,
    down_capture
]
