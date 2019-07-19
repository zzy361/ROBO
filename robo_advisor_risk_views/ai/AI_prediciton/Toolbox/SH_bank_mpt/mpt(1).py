import scipy.optimize as sco
import pandas as pd
import numpy as np
import datetime


def w_adj(w, n, down, up):
    w2 = pd.DataFrame(np.array(w).T, columns=['w']).sort_values(by='w', ascending=0)
    w2.iloc[0:n, 0] = w2.iloc[0:n, 0] * 1 / sum(w2.iloc[0:n, 0])
    w2.iloc[n:, 0] = 0

    j = n - 1
    while w2.iloc[j, 0] < down:
        w2.iloc[0, 0] += w2.iloc[j, 0]
        w2.iloc[j, 0] = 0
        j -= 1

    i = 0
    while w2.iloc[i, 0] > up:
        w2.iloc[i + 1, 0] += w2.iloc[i, 0] - up
        w2.iloc[i, 0] = up
        i += 1
    w3 = np.array(w2.sort_index().iloc[:, 0]).round(4)

    if sum(w3) > 1:
        w3[w3.argmax()] -= sum(w3) - 1
    elif sum(w3) < 1:
        w3[np.where(w3 != 0, w3, w3 + 1).argmin()] += 1 - sum(w3)
    return w3.round(4)


class PercenttileCal:
    def sharpe(self, w):
        today = self.data.index[-1]
        start_date = self.data.index[0]
        year = ((today - start_date).days + 1) / 365
        w = np.array(w)
        fund_temp = np.log(self.data / self.data.shift(1))
        fund_temp = fund_temp.dropna()
        ft_cov = fund_temp.cov()
        rtn = (np.dot(w.T, self.data.iloc[-1] / self.data.iloc[0])) ** (1 / year) - 1

        vol = np.sqrt(np.dot(w.T, np.dot(ft_cov * 252, w)))
        return [rtn, vol, rtn / vol]

    def min_vol(self, w):
        return self.sharpe(w)[1]

    def fractile_cal(self, data, percentlist):
        self.data = data

        today = self.data.index[-1]
        start_date = self.data.index[0]
        year = ((today - start_date).days + 1) / 365
        up_bound = (self.data.iloc[-1] / self.data.iloc[0]).max() ** (1 / year) - 1
        up_bound = up_bound - 0.0005

        n = data.shape[1]
        bnds = tuple((0, 1) for x in range(n))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        minvol = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
        low_bound=self.sharpe(minvol['x'])[0]

        percentlist = low_bound + np.array(percentlist) * (up_bound - low_bound)

        wlist = []
        for tgt in percentlist:
            cons = ({'type': 'eq', 'fun': lambda x: self.sharpe(x)[0] - tgt}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)

            wlist.append(w_adj(res['x'], 5, 0.05, 0.35))

        wlist = np.array(wlist)
        namelist = np.array(self.data.columns)
        idlist,wlist1=[],[]
        for i in range(len(wlist)):
            wlist1.append(wlist[i][wlist[i].nonzero()].tolist())
            idlist.append(namelist[wlist[i].nonzero()].tolist())
        return [idlist,wlist1]


from dateutil.relativedelta import relativedelta




