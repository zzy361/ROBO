
'''
@Time    : 2019/3/5 11:08
@author  : weixiang
@file    : rebalance_config.py
@des:

'''

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from decimal import Decimal
import math
import os
import sys
import logging
import time

poc_list = list(map(lambda x: 'ft' + x, [str(i) for i in list(range(1, 10))]))
risk_level = sorted([0.05, 0.1, 0.15] * 3)
risk_level = dict(zip(poc_list, risk_level))

conns = create_engine(
    'mysql+pymysql://' + os.environ['MYSQL_USER'] + ':' + os.environ['MYSQL_PASSWORD'] + '@' + os.environ[
        'MYSQL_HOST'] + ':' + os.environ['MYSQL_PORT']+'/ra_fttw?charset=utf8')

trading_records = pd.read_sql("select * from ra_fttw.ra_para  ; ", con= conns)

trading_records["rebalance"] = str(risk_level)
trading_records.to_sql(con=conns,name='ra_para',if_exists='replace',index=False)

s = time.strftime("%Y-%m-%d %H:%M:%S")
