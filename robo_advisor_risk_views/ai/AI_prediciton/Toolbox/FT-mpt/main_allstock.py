import numpy as np
import pandas as pd
from mpt_weights_cal_allstock import PercenttileCal
import copy
import time

back_days = 66
big_asset_bound_df = pd.read_excel('big_asset_bound.xlsx')
macro_asset_bounds = {}
def bound_assign(bound_df, clas):
    macro_asset_dict = {}
    for i in bound_df.index[:-1]:
        macro_asset_dict[bound_df.loc[i, '资产大类']] = {'weight_bound': bound_df.loc[i, clas],
                                                     'fund_name': []}

    return macro_asset_dict

clas = 'etf'
if clas == 'fund':
    fund_nav_data = pd.read_csv('SH_BANK_ADNAV_filled.csv', index_col=0)
    fund_info_data = pd.read_excel('SH_BANK_NEW.xlsx', sheetname='Sheet1')

    macro_asset_bounds = bound_assign(big_asset_bound_df, 'fund_stock')
else:
    fund_nav_data = pd.read_csv('SH_BANK_ETF_NAV_filled.csv', index_col=0)
    fund_info_data = pd.read_excel('SH_BANK_ETF_NEW.xlsx', sheetname='基金基本资料表')
    macro_asset_bounds = bound_assign(big_asset_bound_df, 'etf_stock')
fund_nav_data.dropna(axis=1, how='all', inplace=True)
fund_nav_data = fund_nav_data.iloc[-back_days:, :]
fund_nav_data = fund_nav_data.iloc[:, [i for i in range(0, fund_nav_data.shape[1]) if fund_nav_data.iloc[:, i].dropna().index.min() == fund_nav_data.index[0]]]





stocks_fund_data = fund_info_data[~fund_info_data.FUND_ASSET_CLASS_FOCUS.isin(['Fixed Income', 'Money Market'])]

debt_fund_data = fund_info_data[fund_info_data.FUND_ASSET_CLASS_FOCUS.isin(['Fixed Income', 'Money Market'])]

true_list=[]
for i in stocks_fund_data.CLASS:
    if 'EM-' == i:
        true_list.append(True)
    else:
        true_list.append(False)

EM_data = stocks_fund_data[true_list]
stocks_fund_list = list(stocks_fund_data.Bloomberg_Ticker)
EM_list =list(EM_data.Bloomberg_Ticker)
stocks_fund_list = [i for i in stocks_fund_list if i in fund_nav_data.columns]
EM_list = [i for i in EM_list if i in fund_nav_data.columns]

debt_fund_list = list(debt_fund_data.Bloomberg_Ticker)

debt_fund_list = [i for i in debt_fund_list if i in fund_nav_data.columns]
available_fund = copy.deepcopy(stocks_fund_list)

fund_nav_data = fund_nav_data[stocks_fund_list]





macro_asset_dict = {'stocks': stocks_fund_list, 'EM': EM_list}

for i in macro_asset_bounds.keys():
    a=list(fund_info_data[fund_info_data['Category'] == i].Bloomberg_Ticker)
    alist = [i for i in a if i in fund_nav_data.columns]
    macro_asset_bounds[i]['fund_name'] = alist

percenttile_obj = PercenttileCal()
percentile_list = [0.7]
start = time.time()
mpt_weight = percenttile_obj.fractile_cal(fund_nav_data.iloc[-back_days:-1, :], percentile_list, macro_asset_bounds)
end = time.time()
for i in mpt_weight:
    for j in i:
        print(j)


for i in mpt_weight[0]:
    classes = []
    for j in i:
        if j in stocks_fund_list:
            classes.append(0)
        else:
            classes.append(1)
    print(classes)
print('lasting time=', end-start)
