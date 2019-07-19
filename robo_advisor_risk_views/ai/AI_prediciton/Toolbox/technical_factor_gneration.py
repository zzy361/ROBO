''' IMPORTS '''
from sklearn.model_selection import TimeSeriesSplit

import math
from sklearn import svm
import pandas as pd
import datetime

from pandas_datareader import data as web
import datetime as dt
from sklearn import preprocessing

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
import numpy as np
from textwrap import wrap
from scipy.stats import linregress
import matplotlib.dates as mdates
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def get_bollinger_bands(rm, rstd):

    upper_band = rm + (2 * rstd)
    lower_band = rm + (-2 * rstd)
    return upper_band, lower_band


def ConvertSeriesToDf(pdSeries, pLstColumns):

    df = pd.DataFrame(pdSeries).reset_index()
    df.columns = pLstColumns
    return df


def CalculateLogReturn(df_data, N=1):

    return np.log(df_data['Close']).diff()


def CalculateEMV(data, nDays):

    dm = ((data['High'] + data['Low']) / 2) - ((data['High'].shift(1) + data['Low'].shift(1)) / 2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br
    EVM_MA = pd.Series(EVM.rolling(window=nDays).mean(), name='EMV')
    data = data.join(EVM_MA)
    return data


def CalcForceIndex(data, ndays):

    FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name='ForceIndex')
    data = data.join(FI)
    return data


def OBV(df_data, n):


    df_data['OBV'] = 0
    df_data['OBV'] = np.where(df_data['Adj Close'] - df_data['Adj Close'].shift(1) > 0,
                          df_data['Volume'], df_data['OBV'])

    df_data['OBV'] = np.where(df_data['Adj Close'] - df_data['Adj Close'].shift(1) == 0, 0, df_data['OBV'])

    df_data['OBV'] = np.where(df_data['Adj Close'] - df_data['Adj Close'].shift(1) < 0, -df_data['Volume'], df_data['OBV'])


    OBV_ma = df_data['OBV'].rolling(window=n).mean()

    OBV_ma = ConvertSeriesToDf(OBV_ma, ['Date', 'OBV'])


    del df_data['OBV']
    df_data.index.rename('date',inplace=True)
    df_data = df_data.merge(OBV_ma, how='left', on=['Date']).set_index(['Date'], drop=False)

    return df_data


def CalcRSI(price, n=14):

    ''' rsi indicator '''
    gain = (price - price.shift(1)).fillna(0)

    def rsiCalc(p):

        avgGain = p[p > 0].sum() / n
        avgLoss = -p[p < 0].sum() / n
        rs = avgGain / avgLoss
        return 100 - 100 / (1 + rs)


    return gain.rolling(window=n).apply(rsiCalc)


def CalcNDayNetPriceChange(df_data, N=2):

    df_data[str(N) + 'DayNetPriceChange'] = df_data['Adj Close'] - df_data['Adj Close'].shift(N)
    return df_data


def CalcAvgVolumeStats(df_data, N=10):



    rollingAvgVol = df_data['Volume'].rolling(window=N).mean()

    rollingAvgVol = ConvertSeriesToDf(rollingAvgVol, ['Date', 'AvgVolume'])

    df_data = df_data.merge(rollingAvgVol, how='left', on=['Date']).set_index(['Date'], drop=False)

    df_data['DiffercenceBtwnAvgVol'] = df_data['Volume'] - df_data['AvgVolume']




    df_data['UpDownVolumeChange'] = ''
    df_data['UpDownVolumeChange'] = np.where(np.logical_and(df_data['Adj Close'] - df_data['Adj Close'].shift(1) >= 0, df_data['DiffercenceBtwnAvgVol'] > 0),
                                         'UpOnAboveAvg', df_data['UpDownVolumeChange'])

    df_data['UpDownVolumeChange'] = np.where(np.logical_and(df_data['Adj Close'] - df_data['Adj Close'].shift(1) < 0, df_data['DiffercenceBtwnAvgVol'] > 0),
                                         'DownOnAboveAvg', df_data['UpDownVolumeChange'])


    df_data['UpDownVolumeChange'] = np.where(np.logical_and(df_data['Adj Close'] - df_data['Adj Close'].shift(1) < 0, df_data['DiffercenceBtwnAvgVol'] < 0),
                                         'DownOnBelowAvg', df_data['UpDownVolumeChange'])

    df_data['UpDownVolumeChange'] = np.where(np.logical_and(df_data['Adj Close'] - df_data['Adj Close'].shift(1) >= 0, df_data['DiffercenceBtwnAvgVol'] < 0),
                                         'UpOnBelowAvg', df_data['UpDownVolumeChange'])

    le = LabelEncoder()
    le.fit(df_data['UpDownVolumeChange'])

    df_data['UpDownVolumeChange'] = le.transform(df_data['UpDownVolumeChange'])

    return df_data


def Wraplinregress(pValues):

    iLen = len(pValues)
    lXVals = range(iLen)
    pValues = pValues / 1000000

    slope_0, intercept, r_value, p_value, std_err = linregress(lXVals, pValues)
    return slope_0



def Sin(x, freq, amplitude, phase, offset):

    return np.sin(x * freq + phase) * amplitude + offset


def WrapCurve_Fit(pValues, pOutputCol=0):

    try:
        iLen = len(pValues)
        lXVals = np.arange(iLen)

        guess_freq = 1
        guess_amplitude = 3 * np.std(pValues) / (2 ** 0.5)
        guess_phase = 0
        guess_offset = np.mean(pValues)

        p0 = [guess_freq, guess_amplitude,
              guess_phase, guess_offset]


        fit = curve_fit(Sin, lXVals, pValues, p0=p0, maxfev=5500)

        return fit[0][pOutputCol]
    except Exception as e:
        print(str(e))



def moving_average_convergence(df_data, nslow=26, nfast=12):

    df_data['emaslow'] = df_data["Close"].ewm(span=nslow, min_periods=nslow).mean()
    df_data['emafast'] = df_data["Close"].ewm(span=nfast, min_periods=nfast).mean()



    df_data['MACD'] = df_data['emafast'] - df_data['emaslow']

    return df_data


def calculate_slope(df_data, N=10):

    fld = 'Adj Close'

    rollingSineFreq = df_data[[fld]].rolling(window=N).apply(func=WrapCurve_Fit, args=(0,))
    rollingSineFreq = ConvertSeriesToDf(rollingSineFreq, ['Date', 'SineFreq'])


    rollingSineAmp = df_data[[fld]].rolling(window=N).apply(func=WrapCurve_Fit, args=(1,))
    rollingSineAmp = ConvertSeriesToDf(rollingSineAmp, ['Date', 'SineAmp'])


    rollingSinePhase = df_data[[fld]].rolling(window=N).apply(func=WrapCurve_Fit, args=(2,))
    rollingSinePhase = ConvertSeriesToDf(rollingSinePhase, ['Date', 'SinePhase'])


    rollingSineOffset = df_data[[fld]].rolling(window=N).apply(func=WrapCurve_Fit, args=(3,))
    rollingSineOffset = ConvertSeriesToDf(rollingSineOffset, ['Date', 'SineOffset'])


    rollingSlopeClose = df_data[['Close']].rolling(window=N).apply(func=Wraplinregress)
    rollingSlopeClose = ConvertSeriesToDf(rollingSlopeClose, ['Date', 'CloseSlope'])


    rollingSlopeVol = df_data[['Volume']].rolling(window=N).apply(func=Wraplinregress)
    rollingSlopeVol = ConvertSeriesToDf(rollingSlopeVol, ['Date', 'VolumeSlope'])


    rollingSlopeStdDev = df_data[['rollingStdev20']].rolling(window=N).apply(func=Wraplinregress)
    rollingSlopeStdDev = ConvertSeriesToDf(rollingSlopeStdDev, ['Date', 'StdDevSlope'])




    df_data.index.rename('date',inplace=True)
    df_data = df_data.merge(rollingSlopeVol, how='left', on=['Date']).set_index(['Date'], drop=False)
    df_data.index.rename('date', inplace=True)
    df_data = df_data.merge(rollingSlopeClose, how='left', on=['Date']).set_index(['Date'], drop=False)
    df_data.index.rename('date', inplace=True)
    df_data = df_data.merge(rollingSlopeStdDev, how='left', on=['Date']).set_index(['Date'], drop=False)
    df_data.index.rename('date', inplace=True)
    df_data = df_data.merge(rollingSineFreq, how='left', on=['Date']).set_index(['Date'], drop=False)
    df_data.index.rename('date', inplace=True)
    df_data = df_data.merge(rollingSineAmp, how='left', on=['Date']).set_index(['Date'], drop=False)
    df_data.index.rename('date', inplace=True)
    df_data = df_data.merge(rollingSinePhase, how='left', on=['Date']).set_index(['Date'], drop=False)
    df_data.index.rename('date', inplace=True)
    df_data = df_data.merge(rollingSineOffset, how='left', on=['Date']).set_index(['Date'], drop=False)



    return df_data


def compute_candle_stick_pattern(df_data):


    df_data['Color'] = np.where(df_data['Close'] > df_data['Open'], 'WHITE', 'BLACK')

    df_data['RealBody'] = np.where(df_data['Open'] != 0, np.absolute(df_data['Close'] - df_data['Open']) / df_data['Open'] * 1, 0)


    df_data['UpperShadow'] = np.where(df_data['Close'] > df_data['Open'], (df_data['Close'] - df_data['Open']) / (df_data['High'] - df_data['Open']),
                                  (df_data['Open'] - df_data['Close']) / (df_data['High'] - df_data['Close']))

    df_data['UpperShadow'] = df_data['UpperShadow'] * 1
    df_data['UpperShadow'].fillna(value=0, inplace=True)

    df_data['LowerShadow'] = np.where(df_data['Close'] > df_data['Open'], (df_data['Close'] - df_data['Open']) / (df_data['Close'] - df_data['Low']),
                                  (df_data['Open'] - df_data['Close']) / (df_data['Open'] - df_data['Low']))

    df_data['LowerShadow'] = df_data['LowerShadow'] * 1
    df_data['LowerShadow'].fillna(value=0, inplace=True)

    df_data['BarType'] = np.where(np.logical_and(df_data['High'] >= df_data['High'].shift(1), df_data['Low'] >= df_data['Low'].shift(1)),
                              'Up', '')


    df_data['BarType'] = np.where(np.logical_and(df_data['High'] <= df_data['High'].shift(1), df_data['Low'] <= df_data['Low'].shift(1)),
                              'Down', df_data['BarType'])


    df_data['BarType'] = np.where(np.logical_and(df_data['High'] <= df_data['High'].shift(1), df_data['Low'] >= df_data['Low'].shift(1)),
                              'Inside', df_data['BarType'])

    df_data['BarType'] = np.where(np.logical_and(df_data['High'] > df_data['High'].shift(1), df_data['Low'] < df_data['Low'].shift(1)),
                              'Outside', df_data['BarType'])

    le = LabelEncoder()
    le.fit(df_data['Color'])

    df_data['Color'] = le.transform(df_data['Color'])

    le.fit(df_data['BarType'])
    df_data['BarType'] = le.transform(df_data['BarType'])

    return df_data


def feature_calculate(df_data, look_back_days, slope_look_back_days):

    df_data.index = pd.to_datetime(df_data.index)
    df_data['Date'] = df_data.index
    lstCols = None
    new_columns = [i.capitalize() for i in df_data.columns]
    if 'Adj Close' not in new_columns:
        df_data['Adj Close'] = df_data['Close']
        new_columns.append('Adj Close')
        df_data.columns = new_columns
    else:
        pass



    df_data = compute_candle_stick_pattern(df_data)
    df_data['HighLowRange'] = df_data['High'] - df_data['Low']

    df_data = CalcNDayNetPriceChange(df_data, 2)

    df_data = CalcAvgVolumeStats(df_data, look_back_days)

    df_data = moving_average_convergence(df_data)

    df_data['RSI'] = CalcRSI(df_data['Close'])

    df_data = CalculateEMV(df_data, look_back_days)

    df_data = CalcForceIndex(df_data, look_back_days)


    df_data = OBV(df_data, look_back_days)





    rollingMax = df_data['Adj Close'].rolling(window=look_back_days).max()
    rollingMin = df_data['Adj Close'].rolling(window=look_back_days).min()

    rollingMean = df_data['Adj Close'].rolling(window=look_back_days).mean()
    rollingMeanFifty = df_data['Adj Close'].rolling(window=50).mean()


    rollingStdev = df_data['Adj Close'].rolling(window=look_back_days).std()

    rollingMeanFifty.fillna(value=0, inplace=True)
    rollingMean.fillna(value=0, inplace=True)
    rollingStdev.fillna(value=0, inplace=True)

    rollingMax.fillna(value=0, inplace=True)
    rollingMin.fillna(value=0, inplace=True)




    if False:
        upper_band, lower_band = get_bollinger_bands(rollingMean, rollingStdev)


    if False:
        upper_band = ConvertSeriesToDf(upper_band, ['Date', 'upper_band'])

        lower_band = ConvertSeriesToDf(lower_band, ['Date', 'lower_band'])

    rollingStdev = ConvertSeriesToDf(rollingStdev, ['Date', 'rollingStdev20'])
    rollingMean = ConvertSeriesToDf(rollingMean, ['Date', 'rollingMean20'])

    rollingMax = ConvertSeriesToDf(rollingMax, ['Date', 'rollingMax20'])
    rollingMin = ConvertSeriesToDf(rollingMin, ['Date', 'rollingMin20'])

    rollingMeanFifty = ConvertSeriesToDf(rollingMeanFifty, ['Date', 'rollingMean50'])
    df_data.index.rename('date',inplace=True)
    df_data = df_data.merge(rollingMean, how='inner', on=['Date'], right_index=True)
    df_data = df_data.merge(rollingStdev, how='inner', on=['Date'], right_index=True)
    df_data = df_data.merge(rollingMeanFifty, how='inner', on=['Date'], right_index=True)

    df_data = df_data.merge(rollingMax, how='inner', on=['Date'], right_index=True)
    df_data = df_data.merge(rollingMin, how='inner', on=['Date'], right_index=True)




    df_data = calculate_slope(df_data, slope_look_back_days)

    df_data = df_data[pd.notnull(df_data['MACD'])]

    df_data = df_data[pd.notnull(df_data['CloseSlope'])]


    lstSR = lstCols

    if lstSR != None:
        for item in lstSR:

            if item[0] == 'S' or item[0] == 'Min':
                df_data[item].fillna(value=0, inplace=True)
            else:
                df_data[item].fillna(value=10e4, inplace=True)


    if False:
        df_data = df_data.merge(upper_band, how='inner', on=['Date'], right_index=True)
        df_data = df_data.merge(lower_band, how='inner', on=['Date'], right_index=True)

        df_data = df_data[pd.notnull(df_data['upper_band'])]




    df_data.dropna(how='any')

    del df_data['Date']

    return df_data


def get_yahoo_stock_data(start_date, end_date, symbol):




    dateStart = dt.datetime(start_date.year, start_date.month, start_date.day)
    dateEnd = dt.datetime(end_date.year, end_date.month, end_date.day)

    sSymbol = symbol

    data = web.get_data_yahoo(sSymbol, dateStart, dateEnd)

    data['Date'] = data.index

    quotes = data
    if len(quotes) == 0:
        raise SystemExit


    dfQuotes = quotes[['Date', 'Open', 'Close', 'Adj Close', 'High', 'Low', 'Volume']]


    dfQuotes.Date = pd.to_datetime(dfQuotes.Date)
    dfQuotes.sort(['Date'], inplace=True)

    return dfQuotes


def get_natural_log_prices(df_data, Nperiods=1):

    lstFlds = ['Adj Close', 'Close', 'Open', 'High', 'Low']
    for fld in lstFlds:

        df_data[fld] = np.log(df_data[fld] / df_data[fld].shift(periods=Nperiods))

    return df_data


def window_stack(a, stepsize=1, width=3):

    n = a.shape[0]
    return np.hstack(a[i:1 + n + i - width:stepsize] for i in range(0, width))


def organize_data(to_forecast, window, horizon):



    shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - window + 1, window)
    strides = to_forecast.strides + (to_forecast.strides[-1],)
    X = np.lib.stride_tricks.as_strided(to_forecast,
                                        shape=shape,
                                        strides=strides)
    y = np.array([X[i + horizon][-1] for i in range(len(X) - horizon)])
    return X[:-horizon], y


if __name__ == '__main__':
    df_data = pd.read_csv('000001.csv')
    df_data.dropna(axis=0,how='any')
    df = feature_calculate(df_data=df_data, look_back_days=18, slope_look_back_days=18)
    df.to_csv('df.csv',index=False)
    print(df.columns)
