import scipy.optimize as sco
import pandas as pd
import numpy as np
from datetime import datetime


def w_adj(w, n, low_bound, up_bound, macro_asset_bounds):
    w2 = pd.DataFrame(np.array(w).T, columns=['w']).sort_values(by='w', ascending=0)

    up_bound = np.array(up_bound)[w2.index]
    low_bound = np.array(low_bound)[w2.index]

    w2.iloc[0:n, 0] = w2.iloc[0:n, 0] * 1 / sum(w2.iloc[0:n, 0])# 将最小值赋值为0
    w2.iloc[n:, 0] = 0

    j = n - 1
    while w2.iloc[j, 0] < low_bound[j]:
        w2.iloc[0, 0] += w2.iloc[j, 0]
        w2.iloc[j, 0] = 0
        j -= 1

    i = 0
    while w2.iloc[i, 0] > up_bound[i]:
        w2.iloc[i + 1, 0] += w2.iloc[i, 0] - up_bound[i]
        w2.iloc[i, 0] = up_bound[i]
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

    def fractile_cal(self, data, percentlist, macro_asset_bounds):

        self.data = data

        today = self.data.index[-1]
        start_date = self.data.index[0]
        year = ((today - start_date).days + 1) / 365
        up = (self.data.iloc[-1] / self.data.iloc[0]).max() ** (1 / year) - 1
        up = up - 0.0005

        n = data.shape[1]
        up_bound = np.zeros(n)
        for k in macro_asset_bounds.keys():
            up_bound[[list(data.columns).index(k) for k in macro_asset_bounds[k]['fund_name']]] = macro_asset_bounds[k]['weight_bound']
        low_bound = np.zeros(n)
        bnds = tuple((low_bound[x], up_bound[x]) for x in range(n))

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['美国股票']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['美国股票']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['美国投资级及以上债券']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['美国投资级及以上债券']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['美国高收益债券']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['美国高收益债券']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['欧洲股票']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['欧洲股票']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['欧洲债券']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['欧洲债券']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['欧洲高收益债券']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['欧洲高收益债券']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['新兴亚洲股票']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['新兴亚洲股票']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['新兴市场股票']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['新兴市场股票']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['新兴拉美股票']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['新兴拉美股票']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['欧非中东股票']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['欧非中东股票']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['新兴市场债券']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['新兴市场债券']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['新兴亚洲债券']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['新兴亚洲债券']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['全球债券']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['全球债券']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['全球股票']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['全球股票']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['其他债券']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['其他债券']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['其他股票']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['其他股票']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['黄金']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['黄金']['fund_name']]])},
                {'type': 'ineq', 'fun': lambda x: macro_asset_bounds['能源']['weight_bound'] - sum(np.array(x)[[list(data.columns).index(j) for j in macro_asset_bounds['能源']['fund_name']]])}
                )
        minvol = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
        low = self.sharpe(minvol['x'])[0]
        if low < 0:
            low = 0
        percentlist = low + np.array(percentlist) * (up - low)
        wlist = []
        for tgt in percentlist:
            res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)

            wlist.append(w_adj(res['x'], 5, low_bound, up_bound, macro_asset_bounds))
        wlist = np.array(wlist)
        namelist = np.array(self.data.columns)
        idlist, wlist1 = [], []
        for i in range(len(wlist)):
            wlist1.append(wlist[i][wlist[i].nonzero()])
            idlist.append(namelist[wlist[i].nonzero()])#

        wlist1[0] = wlist1[0]
        idlist[0]=idlist[0].tolist()
        wlist1[0]=wlist1[0].tolist()
        return [idlist, wlist1]
