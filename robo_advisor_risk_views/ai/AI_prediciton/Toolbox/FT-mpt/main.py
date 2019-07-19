import numpy as np
import pandas as pd
from mpt_weights_cal import PercenttileCal
import copy
import time
from risk_leval_select import risk_leval_select


back_days = 66
big_asset_bound_df = pd.read_excel('big_asset_bound.xlsx')
macro_asset_bounds = {}


def bound_assign(bound_df, clas):
    macro_asset_dict = {}
    for i in bound_df.index[:-1]:
        macro_asset_dict[bound_df.loc[i, '资产大类']] = {'weight_bound': bound_df.loc[i, clas],
                                                     'fund_name': []}

    return macro_asset_dict


risk_leval = []
risk_name = 'RR4'
leval_name = 'high'
if risk_name == 'RR3':
    risk_leval = [1, 2, 3]
elif risk_name == 'RR4':
    risk_leval = [2, 3, 4]

elif risk_name == 'RR5':
    risk_leval = [3, 4, 5]

percentile_dict = {'RR3': {'low': 0.3, 'middle': 0.5, 'high': 0.8}, 'RR4': {'low': 0.3, 'middle': 0.50, 'high': 0.8}, 'RR5': {'low': 0.3, 'middle': 0.5, 'high': 0.8}}

all_nav_data = pd.read_excel('nav_col_filled.xlsx', index_col=0)
fund_nav_data = risk_leval_select(all_nav_data, risk_leval)
fund_info_data = pd.read_excel('Franklin_Fund_List_0125_carys.xlsx', sheetname='FT平台_FUND CODE', converters={'基金代號': str})

macro_asset_bounds = bound_assign(big_asset_bound_df, 'fund_blend')

fund_nav_data.dropna(axis=1, how='all', inplace=True)
fund_nav_data = fund_nav_data.iloc[-back_days:, :]

fund_nav_data = fund_nav_data.iloc[:, [i for i in range(0, fund_nav_data.shape[1]) if fund_nav_data.iloc[:, i].dropna().index.min() == fund_nav_data.index[0]]]
for i in macro_asset_bounds.keys():
    a = list(fund_info_data[fund_info_data['Category'] == i]['基金代號'])
    alist = [i for i in a if i in fund_nav_data.columns]
    macro_asset_bounds[i]['fund_name'] = alist


percenttile_obj = PercenttileCal()
percentile_list = [percentile_dict[risk_name][leval_name]]
start = time.time()
mpt_weight = percenttile_obj.fractile_cal(fund_nav_data.iloc[-back_days:-1, :], percentile_list, macro_asset_bounds)
end = time.time()
for i in mpt_weight:
    for j in i:
        print(j)

print('lasting time=', end - start)
