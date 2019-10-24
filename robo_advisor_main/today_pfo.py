
from config import g

g.init()
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from weight_1 import weight_1
import json
import numpy as np
import copy
today_str = (datetime.today().date()).strftime('%Y%m%d')
conn = g.db.connect()
conn.execute("delete from ra_fttw.trading_record where trade_date>=" + today_str)
conn.execute("delete from ra_fttw.trading_record_json where trade_date>=" + today_str)
conn.close()

risk_day = (datetime.today().date() - relativedelta(days=120)).strftime('%Y%m%d')

risk_rules_out = pd.read_sql('select risk_date,poc_name,risk_signal,risk_comment from ra_fttw.risk_rules_out where risk_date>=' + risk_day, con=g.db)
risk_rules_out['risk_date'] = pd.to_datetime(risk_rules_out['risk_date'])
risk_rules_out = risk_rules_out.sort_values(['risk_date', 'poc_name'], ascending=False).drop_duplicates(['poc_name'])

rebalance_out = pd.read_sql('select risk_date,poc_name,rebalance_signal,rebalance_comment from ra_fttw.rebalance_out where risk_date>=' + risk_day, con=g.db)
rebalance_out['risk_date'] = pd.to_datetime(rebalance_out['risk_date'])
rebalance_out = rebalance_out.sort_values(['risk_date', 'poc_name'], ascending=False).drop_duplicates(['poc_name'])

daily_out = pd.read_sql('select poc_name,asset_ids,trade_date,weight from ra_fttw.poem_daily_out where trade_date>=' + risk_day, con=g.db)
daily_out['trade_date'] = pd.to_datetime(daily_out['trade_date'])
daily_out = daily_out[daily_out['trade_date'] == daily_out['trade_date'].max()]

trade_record = pd.read_sql('select poc_name,asset_ids,trade_date,weight,comment from ra_fttw.trading_record where trade_date>=' + risk_day, con=g.db)
trade_record['trade_date'] = pd.to_datetime(trade_record['trade_date'])
temp1 = copy.deepcopy(trade_record)
trade_record = trade_record[trade_record['trade_date'] == trade_record['trade_date'].max()]
trade_record = trade_record[trade_record['poc_name'].isin(list(map(lambda x: 'ft' + x, [str(i) for i in list(range(1, 10))])))]



fundlist = str(tuple(list(trade_record['asset_ids'].unique())))
nav0 = pd.read_sql('select bloomberg_ticker,nav_date,nav_cal from jf_data.ft_tw_nav where bloomberg_ticker in ' + fundlist + " and nav_date>=" + risk_day, con=g.db)
nav0['nav_date'] = pd.to_datetime(nav0['nav_date'])
nav0 = pd.pivot_table(nav0, index='nav_date', columns='bloomberg_ticker', values='nav_cal')
nav0.loc[datetime.today()] = np.nan
nav0 = nav0.fillna(method='ffill')
nav = copy.deepcopy(nav0)
nav0 = nav0[nav0.index >= (pd.to_datetime(trade_record['trade_date'].values[0], format='%Y%m%d', errors='ignore') - relativedelta(days=1))]
#

rtn = pd.DataFrame(nav0.iloc[-1, :] / nav0.iloc[0, :])

trade_record = pd.merge(trade_record, rtn, left_on='asset_ids', right_index=True, how='outer')
trade_record.columns = ['poc_name', 'asset_ids', 'trade_date', 'weight', 'comment', 'rtn']
trade_record_nav = pd.DataFrame(columns=['poc_name', 'asset_ids', 'trade_date', 'weight_new', 'comment'])
for poc in list(map(lambda x: 'ft' + x, [str(i) for i in list(range(1, 10))])):
    temp = temp1[temp1['poc_name'] == poc]
    temp = temp[~temp['comment'].isin(list(map(str, list(range(200)))))]
    temp.sort_values('trade_date', ascending=False, inplace=True)
    last_swap_date = temp['trade_date'].values[0]
    nav1 = nav[nav.index >= last_swap_date]
    rtn0 = pd.DataFrame(nav1.iloc[-1, :] / nav1.iloc[0, :])
    last_swap_trade_record = temp[(temp['poc_name']==poc)&(temp['trade_date']==last_swap_date)]
    last_swap_trade_record = pd.merge(last_swap_trade_record, rtn0, left_on='asset_ids', right_index=True, how='inner')
    last_swap_trade_record.columns = ['poc_name', 'asset_ids', 'trade_date', 'weight', 'comment', 'rtn']
    # tmp = trade_record[trade_record['poc_name'] == poc]
    tmp = copy.deepcopy(last_swap_trade_record)
    tmp['part'] = tmp['weight'] * tmp['rtn']
    tmp['weight_new'] = tmp['part'] / (tmp['part'].sum())
    tmp['weight_new'] = weight_1(tmp['weight_new'])
    trade_record_nav = pd.concat([trade_record_nav, tmp[['poc_name', 'asset_ids', 'trade_date', 'weight_new', 'comment']]])
trade_record_nav.columns = ['poc_name', 'asset_ids', 'trade_date', 'weight', 'comment']

today_pfo_out = pd.DataFrame(columns=['poc_name', 'asset_ids', 'trade_date', 'weight', 'comment'])
for poc in list(map(lambda x: 'ft' + x, [str(i) for i in list(range(1, 10))])):
    tmp = trade_record[trade_record['poc_name'] == poc]
    try:
        tmp['comment'] = tmp['comment'].apply(lambda x: int(x))
    except:
        pass

    if isinstance(tmp['comment'].values[0], str) == False and tmp['comment'].values[0] >= g.blackwindow:
        if datetime.today().date().month % 3 == 2 and datetime.today().date().day == 1:
            today_pfo = daily_out[daily_out['poc_name'] == poc]
            today_pfo['comment'] = 'quartly'
        elif int(risk_rules_out[risk_rules_out['poc_name'] == poc]['risk_signal'].values[0]) > 0:
            today_pfo = daily_out[daily_out['poc_name'] == poc]
            today_pfo['comment'] = 'risk: ' + risk_rules_out[risk_rules_out['poc_name'] == poc]['risk_comment'].values[0]
        elif int(rebalance_out[rebalance_out['poc_name'] == poc]['rebalance_signal'].values[0]) > 0:
            today_pfo = daily_out[daily_out['poc_name'] == poc]
            today_pfo['comment'] = 'rebalance: ' + rebalance_out[rebalance_out['poc_name'] == poc]['rebalance_comment'].values[0]
        else:
            today_pfo = trade_record_nav[trade_record_nav['poc_name'] == poc]
            try:
                today_pfo['comment'] = int(trade_record[trade_record['poc_name'] == poc]['comment'].values[0]) + 1
            except:
                today_pfo['comment'] = 0
    else:
        today_pfo = trade_record_nav[trade_record_nav['poc_name'] == poc]
        try:
            today_pfo['comment'] = int(trade_record[trade_record['poc_name'] == poc]['comment'].values[0]) + 1
        except:
            today_pfo['comment'] = 0
    today_pfo_out = pd.concat([today_pfo_out, today_pfo])

today_pfo_out['trade_date'] = datetime.today().date()

today_pfo_out.to_sql('trading_record', if_exists='append', schema='ra_fttw', con=g.db, index=False)

info = pd.read_sql_table('asset_pool', con=g.db, schema='ra_fttw')
pfo_rr = pd.merge(today_pfo_out, info[['FT_Ticker', 'FT_TW_RISK']], left_on='asset_ids', right_on='FT_Ticker', how='left')

poc_id = pd.read_sql_table('poc_id', con=g.db, schema='ra_fttw')

dict = []
j = 0
for poc in list(map(lambda x: 'ft' + x, [str(i) for i in list(range(1, 10))])):
    recomm_guid = poc_id[poc_id['poc_name'] == poc]['risk_id'].values[0]
    data_date = datetime.today().strftime("%Y-%m-%d 00:00:00")
    roi = round(g.rtn[j], 2)
    risk = round(g.vol[j], 2)
    cp = round(roi / risk, 2)
    tmp = pfo_rr[pfo_rr['poc_name'] == poc]
    # rr = round((tmp['weight'] * tmp['FT_TW_RISK']).sum())
    rr = round((tmp['weight'] * tmp['FT_TW_RISK']).sum(), 2)
    create_date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    dict1 = []
    for i in tmp.index:
        dict1.append({"fund_id": str(tmp.loc[i, 'asset_ids']), "weight": str(int(round(tmp.loc[i, 'weight'] * 100)))})
    dict.append({"recomm_guid": str(recomm_guid),
                 "data": {"recomm_guid": str(recomm_guid), "data_date": str(data_date), "roi": roi, "risk": risk, "cp": cp, "rr": rr, "creat_date": str(create_date),
                          "funds": dict1}})
    j += 1

print(json.dumps(dict))

trade_record_json = pd.DataFrame([[datetime.today().date(), json.dumps(dict)]], columns=['trade_date', 'json'])
trade_record_json.to_sql('trading_record_json', if_exists='append', schema='ra_fttw', con=g.db, index=False)
