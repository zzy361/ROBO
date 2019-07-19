import numpy as np
import pandas as pd
from mpt_weights_cal import PercenttileCal
import copy
import time

back_days = 504
fund_nav_data = pd.read_csv('SH_BANK_ADNAV_filled.csv', index_col=0)
fund_nav_data.dropna(axis=1, how='all', inplace=True)
fund_nav_data = fund_nav_data.iloc[-back_days:,:]
fund_nav_data = fund_nav_data.iloc[:, [i for i in range(0, fund_nav_data.shape[1]) if fund_nav_data.iloc[:, i].dropna().index.min() == fund_nav_data.index[0]]]

fund_info_data = pd.read_excel('SH_BANK.xlsx', sheet_name='Sheet1')
fund_info_data = fund_info_data[fund_info_data.DVD_FREQ.isin([np.nan, 'None'])]
stocks_fund_data = fund_info_data[~fund_info_data.FUND_ASSET_CLASS_FOCUS.isin(['Fixed Income', 'Money Market'])]

debt_fund_data = fund_info_data[fund_info_data.FUND_ASSET_CLASS_FOCUS.isin(['Fixed Income', 'Money Market'])]

stocks_fund_list = list(stocks_fund_data.Bloomberg_Ticker)

stocks_fund_list = [i for i in stocks_fund_list if i in fund_nav_data.columns]
debt_fund_list = list(debt_fund_data.Bloomberg_Ticker)

debt_fund_list = [i for i in debt_fund_list if i in fund_nav_data.columns]

available_fund = copy.deepcopy(stocks_fund_list)
available_fund.extend(debt_fund_list)

fund_nav_data = fund_nav_data[available_fund]
macro_asset_dict = {'stocks': stocks_fund_list, 'debts': debt_fund_list}






percenttile_obj = PercenttileCal()
percentile_list = [0.25, 0.5, 0.75]
start = time.time()
mpt_weight = percenttile_obj.fractile_cal(fund_nav_data.iloc[-504:-1, :], percentile_list, macro_asset_dict)
end = time.time()
print(mpt_weight)

for i in mpt_weight[0]:
    classes = []
    for j in i:
        if j in stocks_fund_list:
            classes.append(0)
        else:
            classes.append(1)
    print(classes)
print('lasting time=',end-start)
