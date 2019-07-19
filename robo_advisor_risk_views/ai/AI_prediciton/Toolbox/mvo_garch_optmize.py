"""
@des:
用GARCH计算未来收益率并重新跑MPT 组合

优化速度版
"""
import scipy.optimize as sco
import pandas as pd
import numpy as np
from datetime import datetime
import inspect
import arch
import copy
from dateutil.relativedelta import relativedelta
from Toolbox.para_function import *
def portfolio_vol_rtn(df,weights,start,end):
    df = df/df.iloc[0,:]
    start = list(df.index).index(start)
    end = list(df.index).index(end)
    portfolio = (df*weights).sum(axis=1)
    vol = standard_deviation(portfolio,start,end)
    rtn = annualized_return_ratio(portfolio,start,end)
    return [vol,rtn]

def get_return_data(result):
    """
    输入原始数据，返回原始数据的简单收益矩阵，去掉第一行

    :param result:
    :return:
    """
    temrawData=copy.deepcopy(result)
    items = list(temrawData.columns)
    returndata = pd.DataFrame()

    for it in items:
        returnname=it+"_return"
        temrawData[returnname]=100*temrawData[it].diff()/temrawData[it]
        returndata[it]=temrawData[returnname]
    returndata=returndata.dropna()

    return returndata


def w_adj(w, n, down, up):
    w2 = pd.DataFrame(np.array(w).T, columns=['w']).sort_values(by='w', ascending=0)
    if n>len(w):
        n=len(w)
    w2.iloc[0:n, 0] = w2.iloc[0:n, 0] * 1 / sum(w2.iloc[0:n, 0])
    w2.iloc[n:, 0] = 0

    j = n - 1
    while w2.iloc[j, 0] < down:
        print()
        w2.iloc[0:j, 0] += w2.iloc[j, 0]/j
        w2.iloc[j, 0] = 0
        j -= 1
    n1=len(w2[w2['w'] != 0])

    i = 0
    while w2.iloc[i, 0] > up:
        w2.iloc[i + 1:n1, 0] += (w2.iloc[i, 0] - up)/(n1-i-1)
        w2.iloc[i, 0] = up
        i += 1
    w3 = np.array(w2.sort_index().iloc[:, 0]).round(4)

    if sum(w3) > 1:
        w3[w3.argmax()] -= sum(w3) - 1
    elif sum(w3) < 1:
        w3[np.where(w3 != 0, w3, w3 + 1).argmin()] += 1 - sum(w3)
    return w3.round(4)




class MVO:
    def __init__(self,data,start,end):
        self.years=(end-start).days/365
        data.index=pd.to_datetime(data.index)
        data=data.dropna(axis=1, how='all')

        nav = data
        nav=nav[(nav.index>=start) & (nav.index<=end)]
        nav= nav.iloc[:, [i for i in range(nav.shape[1]) if
                          (nav.iloc[:, i].dropna().index.min() <= (start+relativedelta(days=7))) and
                          (nav.iloc[:, i].dropna().index.max() >= (end-relativedelta(days=7)))]]
        nav=nav.sort_index()
        nav=nav.fillna(method='pad')
        nav=nav.fillna(method='bfill')
        self.nav=nav


        self.raw = data[self.nav.columns]
        self.raw=self.raw.fillna(method='pad').resample('D').fillna(method='pad')[::-30]
        self.raw=self.raw.sort_index()
        self.raw_rtn=(self.raw / self.raw.shift(1)-1)[1:]
        self.start = start
        self.end = end

        self.daily_rtn = (self.nav / self.nav.shift(1)-1).dropna()
        self.pt_corr = np.matrix(self.daily_rtn.corr())
        self.pt_cov = self.daily_rtn.cov()




    def get_GARCH_para(self, p=1, q=1):
        """
        选用不同阶数模型，预测下一期波动率
        :param data: 
        :param p: 
        :param q: 
        :return: 
        """
        df = self.raw_rtn.copy()
        instrument_id = df.columns
        garch_vol = []
        for item in instrument_id:
            item_name = item + "_return"
            df[item_name] = df[item]*1000

            demo = np.array(df[item_name].dropna())

            garch11 = arch.arch_model(demo, p=p, q=q)
            res = garch11.fit(update_freq=10)
            garch_vol.append(np.sqrt(res.conditional_volatility[-1]/1000))
        return garch_vol

    def garch_vol(self, w):
        """

        :param w:
        :return: 
        """
        tem_var=np.matrix(w*self.garch_var)
        vol = np.sqrt((tem_var * self.pt_corr * tem_var.T).tolist()[0][0])
        return vol

    def sharpe(self, w):
        w = np.array(w)
        rtn = (np.dot(w.T, self.nav.iloc[-1] / self.nav.iloc[0])) ** (1/self.years) - 1
        vol = np.sqrt(np.dot(w.T, np.dot(self.pt_cov * 252, w)))
        return [rtn, vol, rtn / vol]

    def min_vol(self, w):
        return self.sharpe(w)[1]


    def max_rtn(self, w):
        w = np.array(w)
        rtn = (np.dot(w.T, self.nav.iloc[-1] / self.nav.iloc[0])) ** (1/self.years) - 1
        return -rtn


    def norm_cal(self, percentlist):


        n = self.nav.shape[1]
        bnds = tuple((0, 1) for x in range(n))
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        wlist = []
        if percentlist==[0]:

            res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons,tol = 10e-4)
            if len(np.nonzero(w_adj(res['x'], 5, 0.05, 0.35))[0]) == 3:
                if len(np.nonzero(w_adj(res['x'], 3, 0.05, 0.5))[0]) == 2:
                    wlist.append(w_adj(res['x'], 2, 0.05, 0.95))
                else:
                    wlist.append(w_adj(res['x'], 3, 0.05, 0.5))
            else:
                wlist.append(w_adj(res['x'], 5, 0.05, 0.35))
        elif percentlist==[1]:

            res = sco.minimize(self.max_rtn, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons,tol = 10e-4)
            if len(np.nonzero(w_adj(res['x'], 5, 0.05, 0.35))[0]) == 3:
                if len(np.nonzero(w_adj(res['x'], 3, 0.05, 0.5))[0]) == 2:
                    wlist.append(w_adj(res['x'], 2, 0.05, 0.95))
                else:
                    wlist.append(w_adj(res['x'], 3, 0.05, 0.5))
            else:
                wlist.append(w_adj(res['x'], 5, 0.05, 0.35))
        else:

            maxrtn = sco.minimize(self.max_rtn, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons,tol = 10e-4)
            temp1 = self.sharpe(maxrtn['x'])
            up_bound = temp1[0]
            self.up_vol_rtn = [temp1[1], temp1[0]]

            minvol = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons,tol = 10e-4)
            temp2 = self.sharpe(minvol['x'])
            low_bound = temp2[0]
            self.low_vol_rtn = [temp2[1], temp2[0]]

            percentlist = low_bound + np.array(percentlist) * (up_bound - low_bound)


            for tgt in percentlist:
                cons1 = cons+[{'type': 'eq', 'fun': lambda x: self.sharpe(x)[0] - tgt}]
                res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons1,tol = 10e-4)

                wlist.append(res['x'])

        wlist = np.array(wlist)
        namelist = np.array(self.nav.columns)
        idlist, wlist1 = [], []
        self.percentile_up_low = []
        for i in range(len(wlist)):
            wlist1.append(wlist[i][wlist[i].nonzero()].tolist())
            idlist.append(namelist[wlist[i].nonzero()].tolist())
            self.percentile_up_low.append(portfolio_vol_rtn(self.nav,wlist[i],self.start,self.end))
        return [idlist, wlist1]


    def class_cal(self, info, up, percentlist):
        '''
        :param nav:净值
        :param info:第一列为id，第二列为type; info的id pool必须包括nav的id pool
        :param up:字典,上限
        :param percentlist: 分位点
        :return:
        '''


        info = info[info.iloc[:, 0].isin(self.nav.columns)]

        n = self.nav.shape[1]
        bnds = [(0,1)]*n


        cons_str = '''[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}'''


        if len(up)!=0:
            uptype=list(up.keys())
            upbound=list(up.values())
            for i in range(0,len(uptype)):
                cons_str+=''',{'type': 'ineq', 'fun': lambda x: upbound['''+str(i)+'''] - sum(np.array(x)[[list(self.nav.columns).index(j) for j in list(info[info.iloc[:, 1] == uptype['''+str(i)+''']].iloc[:, 0])]])}'''
        cons_str+=']'
        cons=eval(cons_str,{'upbound':upbound,'np':np,'info':info, 'uptype':uptype, 'self':self}, None)




        maxrtn = sco.minimize(self.max_rtn, n * [1 / n], method='SLSQP',bounds=bnds, constraints=cons,tol = 10e-4)
        temp1 = self.sharpe(maxrtn['x'])
        up_bound = temp1[0]
        self.up_vol_rtn = (temp1[1],temp1[0])

        print(w_adj(maxrtn['x'],12,0.001,1))


        minvol = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP',bounds=bnds, constraints=cons,tol = 10e-4)
        temp2 = self.sharpe(minvol['x'])
        low_bound = temp2[0]
        self.low_vol_rtn = (temp2[1], temp2[0])

        print(w_adj(minvol['x'], 12, 0.001, 1))

        percentlist = low_bound + np.array(percentlist) * (up_bound - low_bound)


        wlist = []
        for tgt in percentlist:
            cons1 = cons + [{'type': 'eq', 'fun': lambda x: self.sharpe(x)[0] - tgt}]
            res = sco.minimize(self.min_vol, n * [1 / n], method='SLSQP', bounds=bnds, constraints=cons1,tol = 10e-4)
            print(tgt)

            wlist.append(w_adj(res['x'], 3, 0.03, 0.45))


        wlist = np.array(wlist)
        namelist = np.array(self.nav.columns)
        idlist, wlist1 = [], []

        for i in range(len(wlist)):
            wlist1.append(wlist[i][wlist[i].nonzero()].tolist())
            idlist.append(namelist[wlist[i].nonzero()].tolist())
        return [idlist, wlist1]

if __name__ == "__main__":
    import time
    import numpy as np
    st = time.clock()
    print(st)

    raw = pd.read_csv("/Users/mars/JIFU/python project/JFquant/Backtest_2018/data/shbank/active_fund.csv", header=0, index_col=0)
    raw.index = pd.to_datetime(raw.index)
    raw.shape
    raw.head()




    end_date = raw.index[-1]
    start_date = end_date - relativedelta(months=3)
    data = raw[raw.index >= start_date]
    print(len(data))


    data = data.fillna(method="pad")
    data = data.dropna()
    data.tail()
    data.head()


    def cal_dict(d):
        tem_sum = 0
        for key in d.keys():
            tem_sum += d[key]
        return tem_sum


    info_data = pd.read_excel("/Users/mars/JIFU/python project/JFquant/Backtest_2018/data/shbank/info_carys_20180523.xlsx", sheetname="active", header=0)
    info_data.head()
    info_formate = info_data[["Bloomberg_Ticker", "Category"]]
    info_formate.head()


    category_constraint = pd.read_excel("/Users/mars/JIFU/python project/JFquant/Backtest_2018/data/shbank/info_carys_20180523.xlsx", sheetname="constraint", header=0)
    len(category_constraint)
    category_constraint =category_constraint.dropna()

    fund_stock_d = {}
    fund_blend_stock_d = {}

    fund_stock_d.clear()
    fund_blend_stock_d.clear()
    for i in category_constraint.index:
        tem_id = category_constraint.loc[i, "id_keys"]
        fund_stock_d[tem_id] = category_constraint.loc[i, "stock"]*1.75
        fund_blend_stock_d[tem_id] = category_constraint.loc[i, "stock_bond"] *2
    print((fund_blend_stock_d))



    mpt = MVO(raw, 3)

    s = mpt.class_cal(info_formate, fund_blend_stock_d, [0.75])
    duration = round(time.clock() - st, 2)
    print("duration is {0}".format(duration))
    print(s)
    re_dict = {}

    for i in range(len(s[0][0])):
        tem_code = s[0][0][i]
        tem_weight = s[1][0][i]
        sm = info_data[info_data["Bloomberg_Ticker"] == tem_code]["LONG_COMP_NAME"].values[0]
        cm = info_data[info_data["Bloomberg_Ticker"] == tem_code]["Category"].values[0]
        re_dict[tem_code] = [tem_weight, sm, cm]
    print(re_dict)


