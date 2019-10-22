
'''
@file: g.py
@time: 2017/11/24 13:09
@desc:

全局参数

'''
import pandas as pd
import platform
from sqlalchemy import create_engine
import os

class g():
    def init(self):
        self.days=90

        self.db=create_engine('mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+os.environ['MYSQL_HOST']+':'+os.environ['MYSQL_PORT'])
g=g()
