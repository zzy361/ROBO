
'''
@desc:

计算回测结果的类
'''

import pandas as pd
import numpy as np
from scipy import stats
from six import iteritems
import copy
import statsmodels.api as sm
import math
DAILY = 252


class result_analysis(object):

    def __init__(self, **kwargs):
        self.result_cum_rtn = 0
        self.result_ann_rtn = 0
        self.result_ann_vol = 0
        self.result_max_drawdown = 0
        self.data = kwargs.get("raw_data")
        self.result_summary()

    def result_summary(self):
        """
        输入净值曲线
        Example: 
            2015-07-16    9900
            2015-07-17    10000
            2015-07-20    10080
            2015-07-21    10119
            
        :return: 
        """
        if isinstance(self.data, pd.Series):

            result_dict = {}
            if len(self.data) < 2:
                print("too short to calculate")
                return np.nan
            else:
                result_dict["cum_rtn"] = self.cum_return()
                result_dict["cum_rtn_final"] = self.cum_returns_final()
                self.simple_return()
                result_dict["ann_rtn"] = self.annual_return()
                result_dict["ann_vol"] = self.annual_volatility()
                result_dict["mdd"] = self.max_draw_down()
                result_dict["sp_ratio"] = self.sharpe_ratio()               
        else: 
            print("raw data should be pd.Series")
            return np.nan

        return result_dict
    
    def cum_return(self):
        """
        累积收益
        :param rawdata:
        :return:            返回dataframe的累积收益率
        """
        self.cumrtn = 100*self.data[-1]/self.data[0]
        return self.cumrtn

    def cum_returns_final(self):
        """
        返回累积净收益
    
        Returns
        ------- 
        """
        self.cum_rtn_final = self.data[-1] - self.data[0]
        return self.cum_rtn_final

    def simple_return(self):
        self.simple_rtn = 100*self.data.diff() / self.data
        self.simple_rtn[0] = 0
        return self.simple_rtn

    def annual_return(self):
        """
        年平均收益率
        :param returns:
        :param period:
        :return:
        """


        num_years = float(len(self.data)) / DAILY
        cum_returns_final = self.cum_rtn_final/self.data[0]
        self.annual_rtn = (1. + cum_returns_final) ** (1. / num_years) - 1

        return self.annual_rtn

    def annual_volatility(self, period=DAILY):
        """
        计算年化波动率
    
            Annual volatility.
        """


        self.ann_vol = self.simple_rtn.std()*np.sqrt(period)
        return self.ann_vol

    def max_draw_down(self):
        """
        计算最大回撤
        更简便的方案？？ 计算平均每个最大回撤的持续时间和幅度？？
        
        """
    

        tem_min = self.data[0]
        tem_max = self.data[0]
        max_draw = 0
    
        for i in range(len(self.data)):
            if self.data[i] > tem_max:
                tem_max = self.data[i]
            if self.data[i] < tem_min:
                tem_min = self.data[i]
    
            if self.data[i] < tem_max:
                tem_draw = tem_max - self.data[i]
                if tem_draw > max_draw:
                    max_draw = tem_draw
    
        self.mdd = max_draw/tem_max
        return self.mdd

    def sharpe_ratio(self):
        """
        Determines the Sharpe ratio of a strategy.

        """
    

        self.result_sharp = self.annual_rtn / self.ann_vol
        return self.result_sharp        



if __name__ == "__main__":

    raw = pd.read_csv("D:\\python_code\\get_wind_data\\result_data\\000016.SH.csv", header=0, index_col=0)

    
    raw.index = pd.to_datetime(raw.index)

    ann_rtn = raw["close"]

    c = result_analysis(raw_data=ann_rtn)
    re = c.result_summary()
    print(re)



