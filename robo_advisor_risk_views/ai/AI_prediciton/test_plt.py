import pandas as pd
import matplotlib.ticker as ticker

import math
import matplotlib.pyplot as plt
import numpy as np


def test_plt(raw, back, label):

    raw.index = pd.to_datetime(raw.index)
    raw = raw.iloc[-back:, :]

    raw.loc[:, 'label'] = label
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(1, 1, 1)



    ax1.set_title("Prediction Signal")

    raw.to_excel('raw.xlsx')
    ax1.plot(raw["close"], label='Close Price', color='k')
    holdx = []
    holdy = []
    sellx = []
    selly = []
    buyx = []
    buyy = []

    for i in raw.index:

        tem_last = raw.loc[i, "close"]
        tem_signal = raw.loc[i, "label"]
        if tem_signal == 0:
            holdx.append(i)
            holdy.append(tem_last)
        if tem_signal == 1:
            sellx.append(i)
            selly.append(tem_last)
        elif tem_signal == 2:
            buyx.append(i)
            buyy.append(tem_last)
    ax1.plot(holdx, holdy, 'gD', label=u" Hold signal")
    ax1.plot(sellx, selly, 'bD', label=u" Weak signal")
    ax1.plot(buyx, buyy, 'rD', label=u"Strong signal")
    ax1.legend()
    plt.show()



def test_plt_gif(raw, back, label):

    raw.index = pd.to_datetime(raw.index)
    raw = raw.iloc[-back:, :]







    sellx = []
    selly = []
    buyx = []
    buyy = []

    for k in range(int(len(raw.index) / 3)):
        i = raw.index[3 * k]

        fig = plt.figure(figsize=(20, 12))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("Prediction Signal")

        tem_last = raw.loc[i, "close"]
        tem_signal = raw.loc[i, "signal"]

        if tem_signal == -1:
            sellx.append(i)
            selly.append(tem_last)
        elif tem_signal == 1:
            buyx.append(i)
            buyy.append(tem_last)
        ax1.plot(raw.loc[:i, "close"], label='Close Price', color='k')
        ax1.plot(sellx, selly, 'gD', label=u" Sell point")
        ax1.plot(buyx, buyy, 'ro', label=u"Buy point")
        ax1.set_xlim(raw.index[0], raw.index[-1])
        ax1.set_ylim(1600, 4500)
        ax1.legend()

        plt.savefig('fig/' + str(k) + '.png')
