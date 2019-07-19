import scipy.optimize as sco
import pandas as pd
import numpy as np
from datetime import datetime


def w_adj(w, n, down, up, stock_index):
    w2 = pd.DataFrame(np.array(w).T, columns=['w']).sort_values(by='w', ascending=0)
    w2.iloc[0:n, 0] = w2.iloc[0:n, 0] * 1 / sum(w2.iloc[0:n, 0])
    w2.iloc[n:, 0] = 0
    w2_stock = w2[w2.index.isin(stock_index)]
    w2_debt = w2[~w2.index.isin(stock_index)]

    j = w2_stock.shape[0] - 1
    while w2_stock.iloc[j, 0] < down:
        w2_stock.iloc[0, 0] += w2_stock.iloc[j, 0]
        w2_stock.iloc[j, 0] = 0
        j -= 1

    j = w2_debt.shape[0] - 1
    while w2_debt.iloc[j, 0] < down:
        w2_debt.iloc[0, 0] += w2_debt.iloc[j, 0]
        w2_debt.iloc[j, 0] = 0
        j -= 1

    i = 0
    while w2_stock.iloc[i, 0] > up:
        w2_stock.iloc[i + 1, 0] += w2_stock.iloc[i, 0] - up
        w2_stock.iloc[i, 0] = up
        i += 1

    i = 0
    while w2_debt.iloc[i, 0] > up:
        w2_debt.iloc[i + 1, 0] += w2_debt.iloc[i, 0] - up
        w2_debt.iloc[i, 0] = up
        i += 1

    w3_stock = np.array(w2_stock.sort_index().iloc[:, 0]).round(4)
    w3_debt = np.array(w2_debt.sort_index().iloc[:, 0]).round(4)

    if sum(w3_stock) > 0.5:
        w3_stock[w3_stock.argmax()] -= sum(w3_stock) - 0.5
    elif sum(w3_stock) < 0.5:
        w3_stock[np.where(w3_stock != 0, w3_stock, w3_stock + 0.5).argmin()] += 0.5 - sum(w3_stock)

    if sum(w3_debt) > 0.5:
        w3_debt[w3_debt.argmax()] -= sum(w3_debt) - 0.5
    elif sum(w3_debt) < 0.5:
        w3_debt[np.where(w3_debt != 0, w3_debt, w3_debt + 0.5).argmin()] += 0.5 - sum(w3_debt)
    w3 = np.append(w3_stock, w3_debt)
    return w3.round(4)


class PercenttileCal:
    def sharpe(self, w):

        today = datetime.strptime(self.data.index[-1], "%Y-%m-%d")
        start_date = datetime.strptime(self.data.index[0], "%Y-%m-%d")
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

    def fractile_cal(self, data, percentlist, macro_asset_dict):

        self.data = data
        today = datetime.strptime(self.data.index[-1], "%Y-%m-%d")
        start_date = datetime.strptime(self.data.index[0], "%Y-%m-%d")

        year = ((today - start_date).days + 1) / 365
        print(year)
        up_bound = 0.5 * ((self.data[macro_asset_dict['stocks']].iloc[-1] / self.data[macro_asset_dict['stocks']].iloc[0]).max() ** (1 / year) - 1) \
                   + 0.5 * ((self.data[macro_asset_dict['debts']].iloc[-1] / self.data[macro_asset_dict['debts']].iloc[0]).max() ** (1 / year) - 1)
        up_bound = up_bound - 0.0005

        n = data.shape[1]
        bnds = tuple((0, 1) for x in range(n))
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},

                {'type': 'eq', 'fun': lambda x: 0.5 - sum(np.array(x)[[list(data.columns).index(i) for i in macro_asset_dict['debts']]])})
        minvol = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
        low_bound = self.sharpe(minvol['x'])[0]

        percentlist = low_bound + np.array(percentlist) * (up_bound - low_bound)

        wlist = []
        for tgt in percentlist:
            cons = ({'type': 'eq', 'fun': lambda x: self.sharpe(x)[0] - tgt}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},

                    {'type': 'eq', 'fun': lambda x: 0.5 - sum(np.array(x)[[list(data.columns).index(i) for i in macro_asset_dict['debts']]])})
            res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)

            print(res['x'])
            print(sum(np.array(res['x'])[[list(data.columns).index(i) for i in macro_asset_dict['stocks']]]))
            stock_index = [list(data.columns).index(i) for i in macro_asset_dict['stocks']]
            wlist.append(w_adj(res['x'], 8, 0.05, 0.35, stock_index))

        wlist = np.array(wlist)
        namelist = np.array(self.data.columns)
        idlist, wlist1 = [], []
        for i in range(len(wlist)):
            wlist1.append(wlist[i][wlist[i].nonzero()].tolist())
            idlist.append(namelist[wlist[i].nonzero()].tolist())
        return [idlist, wlist1]


from dateutil.relativedelta import relativedelta




