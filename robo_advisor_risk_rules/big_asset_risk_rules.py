import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from risk_ruler import *

import os
database_address = os.environ['MYSQL_HOST']
def risk_main():

    con = create_engine('mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+database_address+':'+os.environ['MYSQL_PORT']+'/ra_fttw?charset=utf8')

    jf_data_con = create_engine('mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+database_address+':'+os.environ['MYSQL_PORT']+'/jf_data?charset=utf8')

    added_big_asset = ['北美股票']
    fund_info = pd.read_sql('select * from fund_global_info', con=jf_data_con)
    trading_record = pd.read_sql('select * from trading_record', con=con)
    trading_record.sort_values(by='trade_date', ascending=True, inplace=True)
    poc_list = list(trading_record['poc_name'].unique())
    risk_table_name = 'risk_out'
    final_risk_table_name = 'risk_rules_out'
    para_table_name = 'ra_para'
    risk_date = datetime.today()
    risk_ruler_obj = risk_ruler()
    poc_list = ['ft'+str(i) for i in range(1, 10)]
    for i in poc_list:
        risk_ruler_obj.risk_signal(added_big_asset=added_big_asset, fund_info=fund_info,
                                   trading_record=trading_record[trading_record['poc_name']==i], con=con, risk_table_name=risk_table_name,
                                   final_risk_table_name=final_risk_table_name, para_table_name=para_table_name,
                                   risk_date=risk_date, poc_name=i)

        risk_ruler_obj.rule_code = ''
        risk_ruler_obj.final_risk_signal = 0
        risk_ruler_obj.poc_name = ''
        risk_ruler_obj.risk_comment = ''

    risk_ruler_obj.write_to_sql(con=con,table_name=final_risk_table_name)

risk_main()
