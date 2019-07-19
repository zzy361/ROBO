
import pandas as pd


def risk_leval_select(total_fund_data, risk_leval):
    risk_leval_data = pd.read_excel('Franklin_Fund_List_0125_carys.xlsx', sheetname='FT平台_FUND CODE',converters={'基金代號':str})
    real_fund_id = list(map(str, list(risk_leval_data[risk_leval_data['RR等級'].isin(risk_leval)]['基金代號'])))
    all_id = list(total_fund_data.columns)
    select_fund_id = list(set(all_id).difference(set(real_fund_id)))
    for i in select_fund_id:
        del total_fund_data[i]
    return total_fund_data
