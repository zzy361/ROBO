import os
import sys
import poem
from up_bound_cal import up_bound_cal
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from config import g
g.init()

today_str=(datetime.today().date()).strftime('%Y%m%d')
conn=g.db.connect()
conn.execute("delete from ra_fttw.poem_daily_out where trade_date>="+today_str)
conn.close()

info=pd.read_sql_table('fund_global_info',con=g.db,schema='jf_data')
info=info[info['FT_TW_Taipei']>0]

fundlist=str(tuple(list(info['FT_Ticker'])))
pre_day=datetime.today().date()-relativedelta(days=g.days)
pre_day_str=pre_day.strftime('%Y%m%d')

nav0=pd.read_sql('select bloomberg_ticker,nav_date,nav_cal from jf_data.ft_tw_nav where bloomberg_ticker in '+fundlist+" and nav_date>="+pre_day_str,con=g.db)
nav0['nav_date']=pd.to_datetime(nav0['nav_date'])
nav0=pd.pivot_table(nav0,index='nav_date',columns='bloomberg_ticker',values='nav_cal')

risk_day = (datetime.today().date() - relativedelta(days=90)).strftime('%Y%m%d')
risk_info = pd.read_sql('select risk_date,asset_name,risk from ra_fttw.risk_out where risk_date>=' + risk_day,
                        con=g.db)
risk_info['risk_date'] = pd.to_datetime(risk_info['risk_date'])
risk_info = risk_info.sort_values(['risk_date', 'asset_name'],ascending=False).drop_duplicates(['asset_name'])[
    ['asset_name', 'risk']]
risk_info.index = risk_info['asset_name']

df=pd.DataFrame(columns=['id'])
for risk in ['low','mid','high']:
    if risk=='low':
        fundlist=list(info[info['FT_TW_RISK'].isin([1,2,3])]['FT_Ticker'])
        nav=nav0[fundlist]
        cons=up_bound_cal(list(info[info['FT_TW_RISK'].isin([1,2,3])]['Category']),risk_info)
        # print(cons)
        per_list=[0.2,0.3,0.4]
    elif risk=='mid':
        fundlist=list(info[info['FT_TW_RISK'].isin([2,3,4])]['FT_Ticker'])
        nav=nav0[fundlist]
        cons = up_bound_cal(list(info[info['FT_TW_RISK'].isin([2,3,4])]['Category']),risk_info)
        per_list=[0.5,0.6,0.7]
    elif risk=='high':
        fundlist = list(info[info['FT_TW_RISK'].isin([2, 3, 4,5])]['FT_Ticker'])
        nav=nav0[fundlist]
        cons = up_bound_cal(list(info[info['FT_TW_RISK'].isin([2,3,4,5])]['Category']),risk_info)
        per_list = [0.6, 0.75, 0.85]

    nav=nav.sort_index()
    nav.dropna(how='any', inplace=True)
    # print(nav.shape)
    rtn_matrix=(nav/nav.shift(20)-1)
    rtn=list(rtn_matrix.mean()*12)

    nav=nav.fillna(method='ffill')

    nav_daily=(nav/nav.shift(1)-1)[1:]
    rho=nav_daily.corr()
    vol=np.diag(nav_daily.cov()*252)

    anual_rtn = (nav.iloc[-1, :] / nav.iloc[0, :]) ** (nav.shape[0] / 252) - 1
    cvar_dict = {}
    cvar_dict['rtn_matrix'] = anual_rtn.values
    cvar_dict['alpha'] = 0.26
    cvar_dict['beta'] = 1.7
    cvar_dict['expected_cvar'] = -0.05
    asset_cons={'list': [], 'indices': [], 'lb': [], 'ub': [],'n_min':[],'n_max':[]}

    asset_cons['list']=list(cons.keys())

    for i in asset_cons['list']:
        asset_cons['indices'].append([nav.columns.tolist().index(j) for j in list(set(info[info['Category']==i]['FT_Ticker'].tolist()).intersection(set(nav.columns.tolist())))])
        asset_cons['lb'].append(0)
        asset_cons['ub'].append(cons[i])
        asset_cons['n_min'].append(0)
        asset_cons['n_max'].append(2)

    results=poem.mvo(nav.columns,rtn,vol,rho,per_list,5,5,0.05,0.35,asset_cons=asset_cons,cvar_dict=cvar_dict)
    results=pd.DataFrame(results)

    results.columns=[risk+'1',risk+'2',risk+'3']
    df=df.merge(results,how='outer',left_index=True,right_index=True)
    # print(df)

df['id']=df.index

out=pd.DataFrame(columns=['poc_name','asset_ids','trade_date','weight','comment'])
for i in range(1,10):
    # print(i)
    tmp=df.iloc[:,[0,i]].dropna()
    tmp.columns=['asset_ids','weight']
    tmp['poc_name']='ft'+str(i)
    tmp['trade_date']=datetime.today().date()
    tmp['comment']='daily'
    out=pd.concat([out,tmp],sort=False)

out.to_sql('poem_daily_out',if_exists='append',schema='ra_fttw',con=g.db,index=False)
