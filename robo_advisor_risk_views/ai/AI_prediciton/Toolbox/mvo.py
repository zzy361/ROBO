import scipy.optimize as sco
import pandas as pd
import numpy as np
import datetime
import inspect
import sys
import ast


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


class MVO:
    def sharpe(self, w):
        today = self.nav.index[-1]
        start_date = self.nav.index[0]
        year = ((today - start_date).days + 1) / 365
        w = np.array(w)
        fund_temp = np.log(self.nav / self.nav.shift(1))
        fund_temp = fund_temp.dropna()
        ft_cov = fund_temp.cov()
        rtn = (np.dot(w.T, self.nav.iloc[-1] / self.nav.iloc[0])) ** (1 / year) - 1
        vol = np.sqrt(np.dot(w.T, np.dot(ft_cov * 252, w)))
        return [rtn, vol, rtn / vol]

    def min_vol(self, w):
        return self.sharpe(w)[1]

    def max_rtn(self,w):
        return -self.sharpe(w)[0]


    def norm_cal(self, data, percentlist):
        self.nav = data

        n = self.nav.shape[1]
        bnds = tuple((0, 1) for x in range(n))
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        wlist = []
        if percentlist==[0]:

            res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
            if len(np.nonzero(w_adj(res['x'], 5, 0.05, 0.35))[0]) == 3:
                if len(np.nonzero(w_adj(res['x'], 3, 0.05, 0.5))[0]) == 2:
                    wlist.append(w_adj(res['x'], 2, 0.05, 0.95))
                else:
                    wlist.append(w_adj(res['x'], 3, 0.05, 0.5))
            else:
                wlist.append(w_adj(res['x'], 5, 0.05, 0.35))
        elif percentlist==[1]:

            res = sco.minimize(self.max_rtn, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
            if len(np.nonzero(w_adj(res['x'], 5, 0.05, 0.35))[0]) == 3:
                if len(np.nonzero(w_adj(res['x'], 3, 0.05, 0.5))[0]) == 2:
                    wlist.append(w_adj(res['x'], 2, 0.05, 0.95))
                else:
                    wlist.append(w_adj(res['x'], 3, 0.05, 0.5))
            else:
                wlist.append(w_adj(res['x'], 5, 0.05, 0.35))
        else:

            maxrtn = sco.minimize(self.max_rtn, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
            up_bound = self.sharpe(maxrtn['x'])[0]

            minvol = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
            low_bound=self.sharpe(minvol['x'])[0]

            percentlist = low_bound + np.array(percentlist) * (up_bound - low_bound)


            for tgt in percentlist:
                cons1 = cons+[{'type': 'eq', 'fun': lambda x: self.sharpe(x)[0] - tgt}]
                res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons1)
                if len(np.nonzero(w_adj(res['x'], 5, 0.05, 0.35))[0]) == 3:
                    if len(np.nonzero(w_adj(res['x'], 3, 0.05, 0.5))[0]) == 2:
                        wlist.append(w_adj(res['x'], 2, 0.05, 0.95))
                    else:
                        wlist.append(w_adj(res['x'], 3, 0.05, 0.5))
                else:
                    wlist.append(w_adj(res['x'], 5, 0.05, 0.35))

        wlist = np.array(wlist)
        namelist = np.array(self.nav.columns)
        idlist, wlist1 = [], []
        for i in range(len(wlist)):
            wlist1.append(wlist[i][wlist[i].nonzero()].tolist())
            idlist.append(namelist[wlist[i].nonzero()].tolist())
        return [idlist, wlist1]


    def class_cal(self, nav, info, type, upbound, percentlist):
        '''
        :param nav:净值
        :param info:第一列为id，第二列为type; info的id pool必须包括nav的id pool
        :param type: 尽量包括所有的分类
        :param upbound: 上限
        :param percentlist: 分位点
        :return:
        '''
        self.nav = nav

        info = info[info.iloc[:, 0].isin(self.nav.columns)]

        n = nav.shape[1]
        bnds = tuple((0, 1) for x in range(n))


        cons_str='''[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}'''
        for i in range(0,len(type)):
            cons_str+=''',{'type': 'ineq', 'fun': lambda x: upbound['''+str(i)+'''] - sum(np.array(x)[[list(self.nav.columns).index(j) for j in list(info[info.iloc[:, 1] == type['''+str(i)+''']].iloc[:, 0])]])}'''
        cons_str+=']'
        cons=eval(cons_str,{'upbound':upbound,'np':np,'info':info,'type':type,'self':self},None)









        maxrtn = sco.minimize(self.max_rtn, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
        up_bound = self.sharpe(maxrtn['x'])[0]
        print(w_adj(maxrtn['x'], 12, 0.05, 1))


        minvol = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons)
        low_bound = self.sharpe(minvol['x'])[0]
        percentlist = low_bound + np.array(percentlist) * (up_bound - low_bound)


        wlist = []
        for tgt in percentlist:
            cons1 = cons + [{'type': 'eq', 'fun': lambda x: self.sharpe(x)[0] - tgt}]
            res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons1)
            print(tgt)

            wlist.append(w_adj(res['x'], 10, 0.01, 0.35))




        wlist = np.array(wlist)
        namelist = np.array(self.nav.columns)
        idlist, wlist1 = [], []
        for i in range(len(wlist)):
            wlist1.append(wlist[i][wlist[i].nonzero()].tolist())
            idlist.append(namelist[wlist[i].nonzero()].tolist())
        return [idlist, wlist1]


