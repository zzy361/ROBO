import pandas as pd
import numpy as np
import math

clas = 'fund'
risk = 'high'
mix = 'stock_bond'
origion_money = 100000
ticker_name = ['TEMTHAI LX Equity', 'MCASEFU TT Equity', 'TEMTECI LX Equity', 'INVPGLI LX Equity', 'ASMA2US LX Equity', 'PRUHYBA LX Equity']
weight_list = [0.2046, 0.1077, 0.0827, 0.105, 0.3247, 0.1753]


info_1 = pd.read_excel('SH_BANK_ETF.xlsx', sheetname='Sheet1')
info_2 = pd.read_excel('SH_BANK_ETF.xlsx', sheetname='Sheet3')


nav = pd.read_csv('SH_BANK_ETF_NAV_filled.CSV', index_col=0)

price_list = list(nav[ticker_name].iloc[-1, :])

isin_list = [info_1[info_1.Bloomberg_Ticker == i]['ID_ISIN'].values[0] for i in ticker_name]
if clas == 'fund':
    cost = 0.01

    name_list = [info_2[info_2.isincode == i]['產品名稱(上海)'].values[0] for i in isin_list]
    pd_result = pd.DataFrame(columns=['Date', 'Name', 'Bloomberg_id', 'ISIN', 'Position', 'NAV', 'Value', 'Weight', 'Fee'])
    for i in range(len(ticker_name)-1):
        position = origion_money * weight_list[i] / ((1+cost) * price_list[i])
        position = math.floor(position * 10000) / 10000
        fee = cost * position * price_list[i]
        fee = math.floor(fee * 100) / 100
        if i == 0:
            pd_result.loc[i] = ['2018/3/1', name_list[i], ticker_name[i], isin_list[i], position, price_list[i], position * price_list[i], weight_list[i], fee]
        else:
            pd_result.loc[i] = [np.nan, name_list[i], ticker_name[i], isin_list[i], position, price_list[i], position * price_list[i], weight_list[i], fee]
    total_value = sum(pd_result.Value)
    total_fee = sum(pd_result.Fee)
    cash = origion_money - total_fee - total_value
    last_position = (origion_money/(1+cost) - sum(pd_result.Value))/price_list[-1]
    last_position = math.floor(last_position * 10000) / 10000
    last_fee = cost * last_position * price_list[-1]
    last_fee = math.floor(last_fee * 100) / 100
    pd_result.loc[len(ticker_name)] = [np.nan, name_list[-1], ticker_name[-1], isin_list[-1], last_position, price_list[-1], last_position * price_list[-1], weight_list[-1], last_fee]
    total_fee +=last_fee

    total_value = sum(pd_result.Value)
    total_value_list = [total_value]
    total_value_list.extend([np.nan] * (len(ticker_name)))

    nav_list = [(total_value / origion_money) * 10]
    nav_list.extend([np.nan] * (len(ticker_name)))

    pd_result.Weight = weight_list
    pd_result.loc[len(ticker_name) + 1] = [np.nan, 'TOTAL', np.nan, np.nan, np.nan, (total_value / origion_money) * 10, total_value, 1, total_fee]
else:
    cost=0.005

    name_list = [info_2[info_2.ticker == i]['Chinese'].values[0] for i in ticker_name]

    isin_list = [info_2[info_2.ticker == i]['上海商銀商品代號'].values[0] for i in ticker_name]
    pd_result = pd.DataFrame(columns=['Date', 'Name', 'Bloomberg_id', '上海商銀商品代號', 'Position', 'NAV', 'Value', 'Weight', 'Fee'])
    for i in range(len(ticker_name)-1):
        position = origion_money * weight_list[i] / ((1+cost) * price_list[i])
        position = math.floor(position * 100) / 100
        fee = cost * position * price_list[i]
        fee = math.floor(fee * 100) / 100
        if i == 0:
            pd_result.loc[i] = ['2018/3/1', name_list[i], ticker_name[i], isin_list[i], position, price_list[i], position * price_list[i], weight_list[i], fee]
        else:
            pd_result.loc[i] = [np.nan, name_list[i], ticker_name[i], isin_list[i], position, price_list[i], position * price_list[i], weight_list[i], fee]

    total_value = sum(pd_result.Value)
    total_fee = sum(pd_result.Fee)
    cash = origion_money - total_fee - total_value
    last_position = (origion_money / (1+cost) - sum(pd_result.Value)) / price_list[-1]
    last_position = math.floor(last_position * 10000) / 10000
    last_fee = cost * last_position * price_list[-1]
    last_fee = math.floor(last_fee * 100) / 100
    pd_result.loc[len(ticker_name)] = [np.nan, name_list[-1], ticker_name[-1], isin_list[-1], last_position, price_list[-1], last_position * price_list[-1], weight_list[-1], last_fee]
    total_fee += last_fee

    total_value = sum(pd_result.Value)
    total_value_list = [total_value]
    total_value_list.extend([np.nan] * (len(ticker_name)))

    nav_list = [(total_value / origion_money) * 10]
    nav_list.extend([np.nan] * (len(ticker_name)))

    pd_result.Weight = weight_list
    pd_result.loc[len(ticker_name) + 1] = [np.nan, 'TOTAL', np.nan, np.nan, np.nan, (total_value / origion_money) * 10, total_value, 1, total_fee]


pd_result.to_excel(clas + '_' + risk + '_' + mix + '_1'+'.xlsx', index=False,encoding='gb18030')
