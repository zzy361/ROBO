
'''
@author: weixiang
@file: wxFinacial_func.py
@time: 2018/1/22 9:55
@desc:

'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import copy
from pandas import Series
from Toolbox.volatility_cal import *
from Toolbox.PCA.PCA_analyse import *




def get_smart_weight(cov_mat, method='risk parity', wts_adjusted=False):
    '''
    功能：输入协方差矩阵，得到不同优化方法下的权重配置
    输入：
        cov_mat  pd.DataFrame,协方差矩阵，index和column均为资产名称
        method  优化方法，可选的有min variance、risk parity、max diversification、equal weight
    输出：
        pd.Series  index为资产名，values为weight
    PS:
        依赖scipy package
    '''

    if not isinstance(cov_mat, pd.DataFrame):
        raise ValueError('cov_mat should be pandas DataFrame！')

    omega = np.matrix(cov_mat.values)


    def fun1(x):

        return np.matrix(x) * omega * np.matrix(x).T

    def fun2(x):
        tmp = (omega * np.matrix(x).T).A1
        risk = x * tmp
        delta_risk = [sum((i - risk) ** 2) for i in risk]
        return sum(delta_risk)

    def fun3(x):
        den = x * omega.diagonal().T
        num = np.sqrt(np.matrix(x) * omega * np.matrix(x).T)
        return num / den


    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    bnds = tuple((0, None) for x in x0)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-20}

    if method == 'min variance':
        res = minimize(fun1, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == 'risk parity':
        res = minimize(fun2, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == 'max diversification':
        res = minimize(fun3, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == 'equal weight':
        return pd.Series(index=cov_mat.index, data=1.0 / cov_mat.shape[0])
    else:
        raise ValueError('method should be min variance/risk parity/max diversification/equal weight！！！')


    if res['success'] == False:
        print(res['message'])

    wts = pd.Series(index=cov_mat.index, data=res['x'])
    if wts_adjusted == True:
        wts = wts[wts >= 0.0001]
        return wts / wts.sum() * 1.0
    elif wts_adjusted == False:
        return wts
    else:
        raise ValueError('wts_adjusted should be True/False！')


def getFrequencyData(rawdata, fre="day"):
    """

    :param rawdata:     index 为时间 datetime格式的pd
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
    return result


def GetReturnData(result):
    """
    输入原始数据，返回原始数据的简单收益矩阵，去掉第一行

    :param result:
    :return:
    """
    temrawData = copy.deepcopy(result)
    items = list(temrawData.columns)
    returndata = pd.DataFrame()

    for it in items:
        returnname = it + "_return"
        temrawData[returnname] = 100 * temrawData[it].diff() / temrawData[it]
        returndata[it] = temrawData[returnname]
    returndata = returndata.dropna()

    return returndata


def GetDownReturnData(result):
    """
    输入原始数据，返回原始数据的简单收益矩阵，去掉第一行

    :param result:
    :return:
    """
    temrawData = copy.deepcopy(result)
    items = list(temrawData.columns)
    returndata = pd.DataFrame()

    for it in items:
        returnname = it + "_return"
        temrawData[returnname] = 100 * temrawData[it].diff() / temrawData[it]
        returndata[it] = [0 if i >= 0 else i for i in temrawData[returnname].values]
    returndata = returndata.dropna()

    return returndata


def sharpRatio(rawdata):
    """

    :param rawdata:
    :return:            返回dataframe的累积收益率
    """

    temrawData = copy.deepcopy(rawdata)
    items = list(temrawData.columns)
    returndata = pd.DataFrame()

    for it in items:
        returnname = it + "_sharp"
        startdate = rawdata.index[0]
        fristdata = rawdata.loc[startdate, it]

        temrawData[returnname] = (temrawData[it] / fristdata) - 1
        returndata[it] = temrawData[returnname]

    return returndata


def sharpPlot(rawdata, datalist):
    """
    累计走势图

    :param rawdata:
    :return:
    """


    s = sharpRatio(rawdata)

    fig = plt.figure(figsize=(15, 10))
    plt.grid()
    ax1 = fig.add_subplot(111)



    for it in datalist:
        ax1.plot(s.index, s[it], label=it)
        ax1.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.5)
    ax1.set_xlabel(u'时间')
    ax1.set_ylabel(u'净值')
    plt.show()


def backtest_weight(testTime, rawData, frequency=3, method="risk parity", fre="day"):
    """

    :param testTime:
    :param rawData:
    :param frequency:
    :param method:
    :param fre:
    :return:
    """

    rawData.index = pd.to_datetime(rawData.index)

    startdate = rawData.index[0]


    testenddate = testTime
    frequencys = frequency
    finaldate = rawData.index[-1]

    fres = fre
    methods = method
    c = {}

    while (testenddate < finaldate):


        testdata = rawData[(rawData.index >= startdate) & (rawData.index <= testenddate)]



        temReturn = GetReturnData(testdata)
        corrdata = temReturn.cov()

        s = get_smart_weight(corrdata, method=method, wts_adjusted=False)
        c[testenddate] = s

        testenddate = testenddate + relativedelta(months=frequencys)
        startdate = startdate + relativedelta(months=frequencys)

    return c


def BackTest(testTime, datalist, rawData, frequency=3, startMoney=1000000, fee=0.001, method="risk parity", fre="day"):
    """

    :param testTime:
                    datetime.datetime(2015, 1, 1) 格式数据
    :param datalist:
                    资产代码 list格式
    :param rawData:
                    时间为索引，每列为基金净值的dataFrame
    :param frequency:
                    调仓频率，1个月为单位
    :param startMoney:
                    初始保证金
    :param fee:     手续费

    :param method:
                    模型方法

    :param fre:
                    计算波动率的时间频率

    :return:
                    返回基金净值和交易日志dataframe

    """


    startdate_time = testTime
    startdate_str = startdate_time.strftime("%Y-%m-%d")
    forwardDate = (min(rawData.index))
    forwardDate_str = (min(rawData.index)).strftime("%Y-%m-%d")

    temDate = startdate_time + relativedelta(months=frequency)


    methods = method
    n = 0
    wholeValue = 0
    cash = 0


    backtestData = copy.deepcopy(rawData[(rawData.index >= startdate_str)])

    for datename in datalist:
        weightNames = datename + "_weight" + "_{0}".format(frequency)
        positionNames = datename + "_position" + "_{0}".format(frequency)

        rcName = datename + "_RC" + "_{0}".format(frequency)

        backtestData[weightNames] = 0
        backtestData[positionNames] = 0
        backtestData[rcName] = 0

    holdvaluename = "HoldValue_{0}".format(frequency)
    cashname = "cash_{0}".format(frequency)
    wholevaluename = "wholeValue_{0}".format(frequency)

    backtestData[holdvaluename] = 0
    backtestData[cashname] = 0
    backtestData[wholevaluename] = 0

    valuelist = []
    cashlist = []
    positionlist = {}
    temweight = pd.Series()
    startweight = pd.Series()

    ison = False

    for i in backtestData.index:

        wholeValue = 0
        temwholevalue = valuelist[-1] + cash if len(valuelist) >= 1 else startMoney


        if i == backtestData.index[0]:
            ison = False



            forwarddata = rawData[(rawData.index >= forwardDate) & (rawData.index <= startdate_str)]


            temreslut = getFrequencyData(forwarddata, fre)
            temReturn = GetReturnData(temreslut)

            forwardcov = temReturn.cov() * 250
            startweight = get_smart_weight(forwardcov, method=methods, wts_adjusted=False)

            for instrument in datalist:
                weightNames = instrument + "_weight" + "_{0}".format(frequency)
                positionNames = instrument + "_position" + "_{0}".format(frequency)
                rcName = instrument + "_RC" + "_{0}".format(frequency)

                weight = (startweight.get(instrument))

                temprice = backtestData.at[i, instrument]
                position = int(temwholevalue * weight / temprice)

                positionlist[instrument] = position

                wholeValue += position * temprice
                backtestData.loc[i, weightNames] = weight
                backtestData.loc[i, positionNames] = position

            print("  frist date {0} weight date {1}".format(i, weight))
            cash = temwholevalue - wholeValue





        elif i == temDate:
            ison = True

            forwardDate = forwardDate + relativedelta(months=frequency)
            forwardDate_str = forwardDate.strftime("%Y-%m-%d")
            startdate_time = temDate
            startdate_str = startdate_time.strftime("%Y-%m-%d")

            temDate = (startdate_time + relativedelta(months=frequency))

            temcalculatedata = rawData[(rawData.index >= forwardDate_str) & (rawData.index <= startdate_str)]

            temreslut = getFrequencyData(temcalculatedata, fre)
            temReturn = GetReturnData(temreslut)

            temdatacov = temReturn.cov() * 250
            temweight = get_smart_weight(temdatacov, method=methods, wts_adjusted=False)


            for instrument in datalist:
                weightNames = instrument + "_weight" + "_{0}".format(frequency)
                positionNames = instrument + "_position" + "_{0}".format(frequency)
                rcName = instrument + "_RC" + "_{0}".format(frequency)

                temprice = backtestData.at[i, instrument]
                weight = (temweight.get(instrument))
                position = int((temwholevalue) * weight / temprice)

                wholeValue += position * temprice
                positionlist[instrument] = position

                backtestData.loc[i, weightNames] = weight
                backtestData.loc[i, positionNames] = position


            cash = temwholevalue - wholeValue

            print("current time {0} ; forward time {1} ;temdate time {2}; weight \n {3} ".format(i, forwardDate, temDate, temweight))







        else:

            for instrument in datalist:
                weightNames = instrument + "_weight" + "_{0}".format(frequency)
                positionNames = instrument + "_position" + "_{0}".format(frequency)
                rcName = instrument + "_RC" + "_{0}".format(frequency)

                temprice = backtestData.at[i, instrument]


                yesterday = (i - relativedelta(days=1))



                currentWeight = (temweight.get(instrument)) if ison else (startweight.get(instrument))



                if positionlist:
                    temposition = positionlist.get(instrument) if positionlist else (temwholevalue * currentWeight) / temprice



                else:
                    temposition = 0
                    print("{0} position is none {1}".format(i, positionlist))

                wholeValue += temprice * temposition

                backtestData.loc[i, weightNames] = currentWeight
                backtestData.loc[i, positionNames] = temposition









        valuelist.append(wholeValue)
        cashlist.append(cash)




    backtestData[holdvaluename] = valuelist
    backtestData[cashname] = cashlist
    backtestData[wholevaluename] = backtestData[holdvaluename] + backtestData[cashname]

    return backtestData


def newBackTest(testTime, datalist, rawData, frequency=3, startMoney=1000000, fee=0.001, method="risk parity", fre="day"):
    """
    将配置权重计算好储存放着，等到回测要用时直接读取即可
    :param testTime:
    :param datalist:
    :param rawData:
    :param frequency:
    :param startMoney:
    :param fee:
    :param method:
    :param fre:
    :return:
    """

    backtestweight = backtest_weight(testTime, rawData, frequency)


    if len(backtestweight) < 1:
        print("get weight error!")
        return

    startdate_str = testTime.strftime("%Y-%m-%d")
    backtestData = copy.deepcopy(rawData[(rawData.index >= startdate_str)])

    for datename in datalist:
        weightNames = datename + "_weight" + "_{0}".format(frequency)
        positionNames = datename + "_position" + "_{0}".format(frequency)


        backtestData[weightNames] = 0
        backtestData[positionNames] = 0


    holdvaluename = "HoldValue_{0}".format(frequency)
    cashname = "cash_{0}".format(frequency)
    wholevaluename = "wholeValue_{0}".format(frequency)

    backtestData[holdvaluename] = 0
    backtestData[cashname] = 0
    backtestData[wholevaluename] = 0

    valuelist = []
    cashlist = []
    positionlist = {}
    ison = False
    temweight = pd.Series()

    for i in backtestData.index:
        datestr = i.strftime("%Y-%m-%d")

        wholeValue = 0
        temwholevalue = valuelist[-1] + cash if len(valuelist) >= 1 else startMoney


        if datestr in backtestweight.keys():

            temweight = backtestweight.get(datestr)


            for instrument in datalist:
                weightNames = instrument + "_weight" + "_{0}".format(frequency)
                positionNames = instrument + "_position" + "_{0}".format(frequency)

                weight = (temweight.get(instrument))

                temprice = backtestData.at[i, instrument]
                position = int(temwholevalue * weight / temprice)
                positionlist[instrument] = position

                wholeValue += position * temprice
                backtestData.loc[i, weightNames] = weight
                backtestData.loc[i, positionNames] = position
                ison = True

            print("current time {0} ; weight \n {1} ".format(i, temweight))

            cash = temwholevalue - wholeValue


        else:

            for instrument in datalist:
                weightNames = instrument + "_weight" + "_{0}".format(frequency)
                positionNames = instrument + "_position" + "_{0}".format(frequency)


                temprice = backtestData.at[i, instrument]
                temposition = positionlist.get(instrument)

                wholeValue += temprice * temposition
                backtestData.loc[i, weightNames] = (temweight.get(instrument))
                backtestData.loc[i, positionNames] = temposition

        valuelist.append(wholeValue)
        cashlist.append(cash)




    backtestData[holdvaluename] = valuelist
    backtestData[cashname] = cashlist
    backtestData[wholevaluename] = backtestData[holdvaluename] + backtestData[cashname]

    return backtestData


def BackTestResult(testTime, datalist, rawData, frequency=3, startMoney=1000000, fee=0.001, method="risk parity", fre="day", departure=0.05):
    """
    设置基金偏离范围，当偏离超过0.05即调整
    :param testTime:
    :param datalist:
    :param rawData:
    :param frequency:
    :param startMoney:
    :param fee:
    :param method:
    :param fre:
    :param departure:

    :return:
    """
    backtestweight = backtest_weight(testTime, rawData, frequency)


    if len(backtestweight) < 1:
        print("get weight error!")
        return

    startdate_str = testTime.strftime("%Y-%m-%d")
    backtestData = copy.deepcopy(rawData[(rawData.index >= startdate_str)])

    for datename in datalist:
        weightNames = datename + "_weight" + "_{0}".format(frequency)
        positionNames = datename + "_position" + "_{0}".format(frequency)


        backtestData[weightNames] = 0
        backtestData[positionNames] = 0


    holdvaluename = "HoldValue_{0}".format(frequency)
    cashname = "cash_{0}".format(frequency)
    wholevaluename = "wholeValue_{0}".format(frequency)

    backtestData[holdvaluename] = 0
    backtestData[cashname] = 0
    backtestData[wholevaluename] = 0

    valuelist = []
    cashlist = []
    positionlist = {}
    ison = False
    temweight = pd.Series()

    for i in backtestData.index:
        datestr = i.strftime("%Y-%m-%d")

        wholeValue = 0
        temwholevalue = valuelist[-1] + cash if len(valuelist) >= 1 else startMoney


        if datestr in backtestweight.keys():

            temweight = backtestweight.get(datestr)


            for instrument in datalist:
                weightNames = instrument + "_weight" + "_{0}".format(frequency)
                positionNames = instrument + "_position" + "_{0}".format(frequency)

                weight = (temweight.get(instrument))

                temprice = backtestData.at[i, instrument]
                position = int(temwholevalue * weight / temprice)
                positionlist[instrument] = position

                wholeValue += position * temprice
                backtestData.loc[i, weightNames] = weight
                backtestData.loc[i, positionNames] = position
                ison = True

            print("current time {0} ; weight \n {1} ".format(i, temweight))

            cash = temwholevalue - wholeValue


        else:

            for instrument in datalist:
                weightNames = instrument + "_weight" + "_{0}".format(frequency)
                positionNames = instrument + "_position" + "_{0}".format(frequency)


                temprice = backtestData.at[i, instrument]
                temposition = positionlist.get(instrument)

                wholeValue += temprice * temposition
                backtestData.loc[i, weightNames] = (temweight.get(instrument))
                backtestData.loc[i, positionNames] = temposition

        valuelist.append(wholeValue)
        cashlist.append(cash)




    backtestData[holdvaluename] = valuelist
    backtestData[cashname] = cashlist
    backtestData[wholevaluename] = backtestData[holdvaluename] + backtestData[cashname]

    return backtestData


def indicatCal(rawdata, colNames, startdate, enddate):
    """
    净值的指标计算

    :param rawdata:
    :param colNames: list of str 要运算矩阵的列
    :param startdate:
    :param enddate:
    :return:            dict
    """
    startdate_str = startdate.strftime("%Y-%m-%d")
    enddate_str = enddate.strftime("%Y-%m-%d")
    result = {}

    if startdate < rawdata.index[0] or enddate > rawdata.index[-1]:
        print("time input error!")

    else:
        for it in colNames:


            selectdata = rawdata[(rawdata.index >= startdate_str) & (rawdata.index <= enddate_str)]
            s = pd.Series(selectdata[it])
            s = s.dropna()
            r = s.diff() / s
            tem = []

            nrow = len(selectdata)


            TotalReturn = round((s[-1] / s[0]) - 1, 6)


            rt = (1 + TotalReturn) ** (252.000000 / nrow) - 1

            annualReturn = round(rt, 6)
            annualVolatility = round(np.sqrt(252) * r.std(), 6)

            SharpRatio = round(annualReturn / annualVolatility, 6)


            tem.append(TotalReturn)
            tem.append(annualReturn)
            tem.append(annualVolatility)
            tem.append(SharpRatio)

            result[it] = tem

    return result
def sum_coef_select(df_data,num):
    sum_para = np.sum(df_data.corr(), axis=1)
    a = np.argsort(sum_para)
    return a[:num].index
def pca_select(df_data,num):
    weights,explanation_factors = sklearn_pca(df_data,num)
    mat_temp = np.sum(weights*explanation_factors,axis=1)
    index = np.argsort(mat_temp)[-1:-(num + 1):-1]
    a = list(df_data.columns[index])
    return a
def best_passive_fund_find(index_data, fund_nav, correlation_dict):
    best_passive_fund = {}
    for i in index_data.columns:
        corr_list = []
        temp_df = pd.merge(index_data[i].to_frame(), fund_nav[correlation_dict[i]], how='inner', left_index=True, right_index=True)
        for j in temp_df.columns[1:]:
            corr_list.append(temp_df.iloc[:, 0].corr(temp_df[j]))
        best_passive_fund[i] = correlation_dict[i][np.argmax(np.array(corr_list))]
    return best_passive_fund

if __name__ == '__main__':
    fund_type = '债券型基金'
    index_data = pd.read_csv("F:\\Backtest_2018\\data\\apollo\\mainland_index.csv")
    index_data = index_data.iloc[-250:, :]
    passive_fund_nav = pd.read_csv("F:\\Backtest_2018\\data\\apollo\\mainland_nav_passive.csv")
    fund_info_data = pd.read_excel('F:\\Backtest_2018\\data\\apollo\\mainland_fund_info.xlsx', sheetname='被动型')
    passive_fund_info = fund_info_data[fund_info_data["category_lv2"].isin(['被动指数型债券基金','被动指数型基金'])]
    passive_fund_info = passive_fund_info[passive_fund_info['category_lv1']==fund_type]
    all_passive_index = list(set(passive_fund_info["跟踪指数代码"].values))
    all_passive_index.sort(key=list(passive_fund_info["跟踪指数代码"].values).index)
    index_data = index_data[all_passive_index]
    index_data = index_data.dropna()
    index_data1 = copy.deepcopy(index_data)

    choosen_index = k_means(index_data1, 2)
    index_data.index = pd.to_datetime(index_data.index)
    index_data = index_data.fillna(method='ffill')
    index_data = index_data[choosen_index]
    temreslut = getFrequencyData(index_data)
    temReturn = GetReturnData(temreslut)
    methods = "risk parity"
    forwardcov = temReturn.cov()
    startweight = get_smart_weight(forwardcov, method=methods, wts_adjusted=False)
    correlation_dict = {}
    for i in all_passive_index:
        correlation_dict[i] = passive_fund_info[passive_fund_info["跟踪指数代码"] == i]["证券代码"].values
    best_passive_fund = best_passive_fund_find(index_data, passive_fund_nav, correlation_dict)
    print(best_passive_fund)
    best_index_fund={}
    for i in best_passive_fund.keys():
        best_index_fund[i] = list(passive_fund_info[passive_fund_info["跟踪指数代码"] == i]["证券代码"].values)
    corresponding_passive_fund = [best_passive_fund[i] for i in choosen_index]
    chinese_name = fund_info_data[fund_info_data['证券代码'].isin(corresponding_passive_fund)]['证券简称'].values
    startweight.index = corresponding_passive_fund
    print(startweight)
    print(best_index_fund)
