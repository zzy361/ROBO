import pandas as pd
import numpy as np
import math
from decimal import Decimal
from datetime import datetime
import os
today = datetime.today()
today = today.strftime('%Y-%m-%d')
clas = 'fund'
risk = 'mid'
mix = 'stock'
origion_money = 100000
ticker_name = ['CACBFUA TT Equity', 'YTNABDA TT Equity', 'SCAAUAA HK Equity', 'CIPETPT LX Equity', 'NNFXCUI LX Equity', 'JPCAAUH LX Equity', 'SKABAUS TT Equity', 'STSMEA1 LX Equity']

weight_list = [0.1436, 0.0589, 0.0878, 0.1037, 0.2243, 0.0753, 0.2051, 0.1013]
if clas == 'fund':
    info_1 = pd.read_excel('SH_BANK_NEW.xlsx')
    info_2 = pd.read_excel('shb_fund_info_mars.xlsx')
    nav = pd.read_csv('SH_BANK_ADNAV_filled.CSV', index_col=0)
    price_list = list(nav[ticker_name].iloc[-1, :])
    isin_list = [info_2[info_2.Bloomberg_Ticker == i]['ID_ISIN'].values[0] for i in ticker_name]

    cost = 0.01

    name_list = [info_2[info_2.Bloomberg_Ticker == i]['產品名稱(上海)'].values[0] for i in ticker_name]
    pd_result = pd.DataFrame(columns=['Date', 'Name', 'Bloomberg_id', 'ISIN', 'Position', 'NAV', 'Value', 'Weight', 'Fee'])
else:
    info_1 = pd.read_excel('SH_BANK_ETF_NEW.xlsx' ,sheetname='基金基本资料表')
    info_2 = pd.read_excel('shb_etf_info_mars.xlsx')
    nav = pd.read_csv('SH_BANK_ETF_NAV_filled.CSV', index_col=0)
    price_list = list(nav[ticker_name].iloc[-1, :])

    isin_list = [info_2[info_2.Bloomberg_Ticker == i]['上海商銀商品代號'].values[0] for i in ticker_name]
    print(isin_list)
    cost = 0.005

    name_list = [info_2[info_2.Bloomberg_Ticker == i]['產品名稱(上海)'].values[0] for i in ticker_name]

    pd_result = pd.DataFrame(columns=['Date', 'Name', 'Bloomberg_id', '上海商銀商品代號', 'Position', 'NAV', 'Value', 'Weight', 'Fee'])

for i in range(len(ticker_name) - 1):
    position = origion_money * weight_list[i] / ((1 + cost) * price_list[i])
    position = math.floor(position * 10000) / 10000
    value = position * price_list[i]
    value = math.floor(value * 100) / 100
    fee = round(Decimal(value * cost), 2)

    if i == 0:
        pd_result.loc[i] = [today, name_list[i], ticker_name[i], isin_list[i], position, price_list[i], value, weight_list[i], fee]
    else:
        pd_result.loc[i] = [np.nan, name_list[i], ticker_name[i], isin_list[i], position, price_list[i], value, weight_list[i], fee]
total_value = sum(pd_result.Value)
total_fee = sum(pd_result.Fee)
cash = Decimal(origion_money) - total_fee - Decimal(total_value)

last_position = (Decimal(origion_money) - Decimal(sum(pd_result.Value)) - sum(pd_result.Fee)) / Decimal((1 + cost) * price_list[-1])
last_position = math.floor(last_position * 10000) / 10000

last_value = last_position * price_list[-1]
last_value = math.floor(last_value * 100) / 100

last_fee = cost * last_value
last_fee = round(Decimal(last_fee), 2)
pd_result.loc[len(ticker_name)] = [np.nan, name_list[-1], ticker_name[-1], isin_list[len(ticker_name)-1], last_position, price_list[-1], last_value, weight_list[-1], last_fee]
total_fee += last_fee

total_value = sum(pd_result.Value)
total_value_list = [total_value]
total_value_list.extend([np.nan] * (len(ticker_name)))

nav_list = [(total_value / origion_money) * 10]
nav_list.extend([np.nan] * (len(ticker_name)))

pd_result.Weight = pd_result.Value / total_value

pd_result.loc[len(ticker_name) + 1] = [np.nan, 'TOTAL', np.nan, np.nan, np.nan, (total_value / origion_money) * 10, total_value, 1, total_fee]



from openpyxl import load_workbook
if os.path.exists(today+'.xlsx'):
    book = load_workbook(today + '.xlsx')
    writer = pd.ExcelWriter(today + '.xlsx', engine='openpyxl')
    writer.book = book
    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)

    pd_result.to_excel(writer, sheet_name=clas + '_' + risk + '_' + mix, index=False)
    writer.close()
else:
    pd_result.to_excel(today + '.xlsx', index=False, sheet_name=clas + '_' + risk + '_' + mix)
