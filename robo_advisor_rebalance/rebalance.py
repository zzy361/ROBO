
'''
@Time    : 2019/2/28 17:19
@author  : weixiang
@file    : rebalance.py
@des:

rebalance:

2、不定期：drift,tolance
3、将参数外置

'''
from sqlalchemy import create_engine
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
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import copy

class RebalanceTest():

    def __init__(self):
        self.conns = create_engine(
    'mysql+pymysql://' + os.environ['MYSQL_USER'] + ':' + os.environ['MYSQL_PASSWORD'] + '@' + os.environ[
        'MYSQL_HOST'] + ':' + os.environ['MYSQL_PORT']+'/ra_fttw?charset=utf8')
        self.poc_list = list(map(lambda x: 'ft' + x, [str(i) for i in list(range(1, 10))]))

        self.load_para()

    def load_para(self):
        """
        读取参数
        :return:
        """
        quers = 'SELECT * FROM ra_fttw.ra_para;'
        tem_data= pd.read_sql(quers, con= self.conns)
        date = tem_data["date"].max()
        sds = tem_data[tem_data['date']==date]["rebalance"].values[0]
        self.risk_level = eval(sds)

    def unperiod_rebalance(self):
        """
        poem和trading_record的差异，计算drift
        一般来说，更新poem后就会改变trading_record
       :return:
        """

        self.trading_records = pd.read_sql("select * from ra_fttw.trading_record  ; ", con= self.conns)
        self.poem_daily_out = pd.read_sql("select * from ra_fttw.poem_daily_out  ; ", con= self.conns)

        s = self.trading_records['trade_date'].max()
        p = self.poem_daily_out['trade_date'].max()
        report = {'risk_date':[],"poc_name":[],'rebalance_signal':[],'rebalance_comment':[],'stock_bond_drift':[]}

        for it in self.poc_list:
            tem_rebalance_signal = 0
            report['risk_date'].append(p)
            report["poc_name"].append(it)
            temp = copy.deepcopy(self.trading_records)
            temp = temp[temp['poc_name'] == it]
            temp = temp[~temp['comment'].isin(list(map(str,list(range(160)))))]
            temp.sort_values('trade_date',ascending=False,inplace=True)
            last_swap_date = temp['trade_date'].values[0]
            print(last_swap_date)
            tem_trading_records = self.trading_records[
                (self.trading_records['trade_date'] == s) & (self.trading_records['poc_name'] == it)]
            tem_poem_daily_out = self.trading_records[
                (self.trading_records['trade_date'] == last_swap_date) & (self.trading_records['poc_name'] == it)]
            # print(tem_trading_records)
            # print(tem_poem_daily_out)
            trading_report = self.drift_band(tem_trading_records)
            # print(trading_report)
            poem_report = self.drift_band(tem_poem_daily_out)
            # print(poem_report)

            temdrifts = (abs(trading_report["stock"] - poem_report["stock"]) + abs(
                trading_report["bond"] - poem_report["bond"])) / 2
            report['stock_bond_drift'].append(temdrifts)

            merge_data = pd.merge(tem_trading_records, tem_poem_daily_out, how='outer',
                                  on=['asset_ids'])
            merge_data.fillna(0, inplace=True)
            merge_data['drift'] = abs(merge_data['weight_y'] - merge_data['weight_x'])
            tem_drift = merge_data['drift'].sum()/2
            print(it,tem_drift)
            comments = str()
            tem_rebalance_signal = str()

            if tem_drift >= self.risk_level[it]:
                tem_rebalance_signal = str(1)

                comments = 'drift {0} > {1}'.format(tem_drift, self.risk_level[it])
                if temdrifts >= self.risk_level[it]:
                    tt = ' &&  stock bond drift {0} > 0.05'.format(temdrifts)
                    # comments += tt
                    # tem_rebalance_signal += str(2)
                report['rebalance_comment'].append(comments)
                report['rebalance_signal'].append(tem_rebalance_signal)
            else:

                report['rebalance_signal'].append(str(0))
                report['rebalance_comment'].append('none')


        return report

    def drift_band(self, df):
        """
        计算股债比例
        :return:
        """
        querys = 'SELECT * FROM ra_fttw.asset_pool;'
        infoms = pd.read_sql(querys, con= self.conns)
        assetlist = df["asset_ids"].tolist()
        report = {"stock":0,"bond":0}
        for it in assetlist:
            if it in infoms["FT_Ticker"].tolist():
                item_class = infoms[infoms["FT_Ticker"]==it]['Asset'].values[0]

                if item_class == '股票':
                    report["stock"] += df[df["asset_ids"]==it]["weight"].values[0]
                else:
                    report["bond"] += df[df["asset_ids"]==it]["weight"].values[0]
            else:
                report["bond"] += df[df["asset_ids"] == it]["weight"].values[0]
        return report

    def up_load_sql(self):
        """

        :return:
        """
        pass

    def test(self):
        """
        调用定期，不定期，其他多种方式的再平衡
        :return:
        """

        re = self.unperiod_rebalance()
        print(re)
        rs = pd.DataFrame.from_dict(re)
        rs['iid'] = rs.index
        rs.drop(columns=['stock_bond_drift'],inplace=True)
        rs.to_sql(con=self.conns,name='rebalance_out',if_exists='append', index=False)
        rs.to_csv("result.csv",index=False)
        print(rs.head())

c = RebalanceTest()
c.load_para()
print(c.test())
