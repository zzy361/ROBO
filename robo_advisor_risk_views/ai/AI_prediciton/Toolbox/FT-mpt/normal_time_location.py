import pandas as pd
import numpy as np
import math
from decimal import Decimal
from datetime import datetime

today = datetime.today()
today = today.strftime('%Y-%m-%d')
origin_sheets = pd.read_excel('2018-03-02.xlsx', sheetname=None)
for i in origin_sheets.keys():
    if 'etf' in i:
        price = pd.read_csv('SH_BANK_ETF_NAV_filled.CSV').fillna(method='ffill')
    else:
        price = pd.read_csv('SH_BANK_ADNAV_filled.CSV').fillna(method='ffill')

    a = price[origin_sheets[i].Bloomberg_id[:-1]].iloc[-1,:].values
    origin_sheets[i].NAV = np.append(price[origin_sheets[i].Bloomberg_id[:-1]].iloc[-1,:].values,np.nan)

    origin_sheets[i]['Value'] = origin_sheets[i]['Position'] * origin_sheets[i]['NAV']
    origin_sheets[i].iloc[-1, 6] = 0
    origin_sheets[i]['Value'] = origin_sheets[i]['Value'].map(lambda x: math.floor(x * 100) / 100)
    origin_sheets[i].iloc[-1, 6] = sum(origin_sheets[i].iloc[:-1, 6])

    origin_sheets[i].iloc[-1, 5] = origin_sheets[i].iloc[-1, 6] / 10000

    origin_sheets[i]['Weight'] = origin_sheets[i]['Value'] / origin_sheets[i]['Value'].tolist()[-1]
    origin_sheets[i].iloc[:, 8] = [0] * len(origin_sheets[i])
    origin_sheets[i].iloc[0, 0] = today

writer = pd.ExcelWriter(today+'.xlsx', engine='xlsxwriter')
for i in origin_sheets.keys():
    origin_sheets[i].to_excel(writer, sheet_name=i, index=False)
writer.save()

