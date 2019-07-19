import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Toolbox.apollo.risk_parity import *
from Toolbox.mvo_garch_optmize import MVO
from Toolbox.para_function import *
def portfolio_vol_rtn(df,weights,start,end):
    df = df/df.iloc[0,:]
    start = list(df.index).index(start)
    end = list(df.index).index(end)
    portfolio = (df*weights).sum(axis=1)
    vol = standard_deviation(portfolio,start,end)
    rtn = annualized_return_ratio(portfolio,start,end)
    return [vol,rtn]
def risk_parity_location(df,risk_parity_weights,start,end):
    mvo = MVO(df,start,end)
    mvo.norm_cal([0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    percentile_point = mvo.percentile_up_low
    upper_point = mvo.up_vol_rtn
    low_point = mvo.low_vol_rtn
    percentile_point.extend([upper_point, low_point])
    risk_parity_point = portfolio_vol_rtn(df,risk_parity_weights,start,end)
    com_array=np.array(percentile_point)
    plt.scatter(com_array[:,0],com_array[:,1],c='red')
    plt.scatter(risk_parity_point[0],risk_parity_point[1],c='blue')
    plt.show()
if __name__ =="__main__":
    back_days = 250
    fund_type = '股票型基金'
    index_data = pd.read_csv("F:\\Backtest_2018\\data\\apollo\\mainland_index.csv",index_col=0)
    index_data = index_data.iloc[-back_days:, :]
    passive_fund_nav = pd.read_csv("F:\\Backtest_2018\\data\\apollo\\mainland_nav_passive.csv")
    fund_info_data = pd.read_excel('F:\\Backtest_2018\\data\\apollo\\mainland_fund_info.xlsx', sheetname='被动型')
    passive_fund_info = fund_info_data[fund_info_data["category_lv2"].isin(['被动指数型债券基金', '被动指数型基金'])]
    passive_fund_info = passive_fund_info[passive_fund_info['category_lv1'] == fund_type]
    all_passive_index = list(set(passive_fund_info["跟踪指数代码"].values))
    all_passive_index.sort(key=list(passive_fund_info["跟踪指数代码"].values).index)
    index_data = index_data[all_passive_index]
    index_data = index_data.dropna()
    index_data1 = copy.deepcopy(index_data)

    choosen_index = k_means(index_data1, 4)

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
    corresponding_passive_fund = [best_passive_fund[i] for i in choosen_index]
    chinese_name = fund_info_data[fund_info_data['证券代码'].isin(corresponding_passive_fund)]['证券简称'].values
    startweight.index = corresponding_passive_fund
    print(startweight)
    index_data = index_data[list(best_passive_fund.keys())]

    risk_parity_weights = startweight.values
    start = index_data.index[-back_days]
    end = index_data.index[-1]
    risk_parity_location(index_data,risk_parity_weights, start,end)




