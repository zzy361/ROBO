import numpy as np
import pymysql
from scipy import stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch
from sklearn import metrics


def predict(origional,n ):
    origional_data.index = pd.to_datetime(origional_data.index).fillna(method="bfill")


    origional_data["returnData"] = origional_data.iloc[1:, 0] - origional_data.iloc[:, 0].shift()
    data = np.array(origional_data['returnData'].dropna())


    t = sm.tsa.stattools.adfuller(data)




    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(data, lags=30, ax=ax1)
    plt.grid()


    order = (3, 0)
    model = sm.tsa.ARMA(data, order).fit()


    at = data - model.fittedvalues
    at2 = np.square(at)

    plt.figure(figsize=(15, 5))
    plt.subplot(211)
    plt.plot(at, label='residual')
    plt.legend()
    plt.subplot(212)
    plt.plot(at2, label='residual^2')
    plt.legend(loc=0)


    m = 30
    acf, q, p = sm.tsa.acf(at2, nlags=m, qstat=True)
    out = np.c_[range(1, m + 1), acf[1:], q, p]
    output = pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
    output = output.set_index('lag')
    output




    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(111)
    fig = sm.graphics.tsa.plot_pacf(at2, lags=30, ax=ax1)


    train = data[:-5]
    test = data[-5:]


    am = arch.arch_model(train, mean='AR', lags=5, vol='ARCH', p=16)
    res = am.fit()
    res.summary()
    res.hedgehog_plot()

    res.plot()
    plt.grid()
    len(train)


    pre = res.forecast(horizon=8, start=1080, align="target", method="analytic", simulations=1000)
    pre = res.forecast(horizon=10, start=1080)



    pre.variance[1080:].plot()
    s = pre.variance[1080:]
    s.tail()
    s.head()

    m = 10 * pre.variance.iloc[-1]

    plt.figure(figsize=(15, 5))
    plt.plot(test, label='realValue')
    m.plot(label='predictValue')

    plt.plot(np.zeros(10), label='zero')
    plt.legend(loc=0)




    train = data[:-10]
    test = data[-10:]
    am = arch.arch_model(train, mean='AR', lags=5, vol='GARCH')
    res = am.fit()
    res.summary()
    res.params

    res.plot()
    plt.plot(data)

    res.hedgehog_plot()

    ini = res.resid[-8:]
    a = np.array(res.params[1:9])
    w = a[::-1]
    for i in range(10):
        new = test[i] - (res.params[0] + w.dot(ini[-8:]))
        ini = np.append(ini, new)

    at_pre = ini[-10:]
    at_pre2 = at_pre ** 2
    at_pre2


    ini2 = res.conditional_volatility[-2:]

    for i in range(10):
        new = 0.000007 + 0.1 * at_pre2[i] + 0.88 * ini2[-1]
        ini2 = np.append(ini2, new)
    vol_pre = ini2[-10:]
    plt.figure(figsize=(15, 5))
    plt.plot(data, label='origin_data')
    plt.plot(res.conditional_volatility, label='conditional_volatility')
    x = range(3828 - 10, 3828)
    plt.plot(x, vol_pre, '.r', label='predict_volatility')
    plt.grid()
    plt.legend(loc=0)
    plt.show()
