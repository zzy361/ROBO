
'''
@file: g.py
@time: 2017/11/24 13:09
@desc:

全局参数

'''
import pandas as pd
import numpy as np
import platform
from sqlalchemy import create_engine
import os

class g():
    def init(self):
        self.days=60

        self.db=create_engine('mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+os.environ['MYSQL_HOST']+':'+os.environ['MYSQL_PORT'])
        self.rtn = list(np.linspace(3, 8, 9))
        self.vol = list(np.linspace(3.7, 8.7, 9))
        self.blackwindow = 15

g=g()
