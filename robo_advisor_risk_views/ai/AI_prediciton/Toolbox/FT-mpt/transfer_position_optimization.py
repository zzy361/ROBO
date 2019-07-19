import pandas as pd
import numpy as np
from datetime import datetime
import math
from decimal import Decimal
today = datetime.today()
today = today.strftime('%Y-%m-%d')
former_pos_info = pd.read_excel('2018-03-06.xlsx', sheetname=None)

new_pos_info = pd.read_excel('2018-03-29.xlsx', sheetname=None)


origion_money = 90000
etf_fee = 0.005
fund_fee = 0.01
cost = 0
for i in new_pos_info.keys():
    if 'fund' in i and 'bond' in i:
        company_info = pd.read_excel('shb_fund_info_mars.xlsx')
        cost = fund_fee
        former_merge = pd.merge(former_pos_info[i], company_info, how='inner', left_on='Bloomberg_id', right_on='Bloomberg_Ticker')
        new_merge = pd.merge(new_pos_info[i], company_info, how='inner', left_on='Bloomberg_id', right_on='Bloomberg_Ticker')
        former_weight = former_merge.groupby(['FUND_COMP_NAME'])['Weight'].sum()
        new_weight = new_merge.groupby(['FUND_COMP_NAME'])['Weight'].sum()
        same_company = [i for i in former_weight.index if i in new_weight.index]
        for j in new_pos_info[i].index:
            position = origion_money * new_pos_info[i].loc[j,'Weight'] / ((1 + cost) * new_pos_info[i].loc[j,'NAV'])
            new_pos_info[i].loc[j, 'Position'] = math.floor(position * 10000) / 10000
            value = position * new_pos_info[i].loc[j, 'NAV']
            new_pos_info[i].loc[j, 'Value'] = math.floor(value * 100) / 100
            fee = round(Decimal(value * cost), 2)
            new_pos_info[i]['Fee'] = fee
        if len(same_company) != 0:
            weight_dis = former_weight[same_company] - new_weight[same_company]
            for k in same_company:
                new_pos_info[i][new_pos_info[i]['Bloomberg_id'].isin(new_merge[new_merge['FUND_COMP_NAME'] == k]['Bloomberg_Ticker'])]['Fee'] = origion_money * weight_dis[k] / (1 + cost)

        else:
            pass
    else:
        company_info = pd.read_excel('shb_etf_info_mars.xlsx')
        cost = etf_fee

writer = pd.ExcelWriter('a' + '.xlsx', engine='xlsxwriter')
for i in new_pos_info.keys():
    new_pos_info[i].to_excel(writer, sheet_name=i, index=False)
writer.save()
