
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sqlalchemy import create_engine
import scipy.stats as st
import warnings
import datetime
import os
import talib as tb
import time

dir=os.path.dirname(os.path.realpath(__file__))
warnings.filterwarnings('ignore')
df_config=pd.read_csv(dir+'/Config_Multifactors.csv')

aws='mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+os.environ['MYSQL_HOST']+':'+os.environ['MYSQL_PORT']
aliyun='mysql+pymysql://tommy:tommy@wei@rm-uf679020c6vrt28in7o.mysql.rds.aliyuncs.com:3306/jf_data?charset=utf8'
aws=aws

class multifactor_asset():
    def __init__(self):
        self.conn =create_engine(aws)

    def Date_adj(self,data1, data2):
        date_nav = data1.nav_date
        temp = pd.merge(date_nav, data2, on='nav_date', how='outer')
        temp.sort_values(by='nav_date', inplace=True)
        temp.fillna(method='ffill', inplace=True)
        temp = pd.merge(date_nav, temp, on='nav_date', how='left')
        return temp

    def Pct_shift(self,data_, list_columns, list_days):
        data_raw = data_.copy()
        list_col = ['nav_date']
        for i in (list_columns):
            for days in list_days:
                Col_name = i + '_' + str(days)
                list_col += [Col_name]
                if days == 0:
                    data_raw[Col_name] = data_raw[i]
                else:
                    data_raw[Col_name] = data_raw[i].shift(days)
                    data_raw[Col_name] = data_raw[i] / data_raw[Col_name] - 1

        return data_raw[list_col]

    def MaxMinNormalization(self,x):
        list_s = list(x.columns)
        list_s.remove('nav_date')
        for Col_name in list_s:

            max = x[Col_name].max()
            min = x[Col_name].min()
            x[Col_name] = (x[Col_name] - min) / (max - min)
            x[Col_name] = st.norm.ppf(x[Col_name])
            x.replace(-np.inf, -3, inplace=True)
            x.replace(np.inf, 3, inplace=True)
        return x

    def run_ols(self):
        df_result = pd.DataFrame()
        for benchmark in df_config['Y'].unique():
            df_sql = df_config[df_config['Y'] == benchmark][['Y', 'Exp', 'Config', 'Table', 'X']].copy(0)
            df_sql['Exp_adj'] = list(map(lambda x: ('%%').join(x.split('%')), df_sql['Exp']))
            query = '''
            SELECT nav_date,close  FROM jf_data.index_ohlcv_pe where bloomberg_ticker =\'{0}\' and nav_date>'20060101'  order by  nav_date asc
            '''.format(benchmark)
            df_y = pd.read_sql(query, self.conn)
            df_y.rename(columns={"close": benchmark}, inplace=True)
            df_y.dropna(inplace=True)
            df_technic=df_y.copy()

            list_technic=['MA','ROC','CCI','RSI','STD','ANGLE']
            df_technic['MA']=tb.MA(df_technic[benchmark],5)/tb.MA(df_technic[benchmark],40)
            df_technic['ROC'] = tb.ROC(df_technic[benchmark],20)
            df_technic['CCI'] = tb.CCI(df_technic[benchmark],df_technic[benchmark],df_technic[benchmark], 20)
            df_technic['RSI'] = tb.RSI(df_technic[benchmark], 20)
            df_technic['STD'] = tb.STDDEV(df_technic[benchmark], 20)
            df_technic['ANGLE']=tb.LINEARREG_ANGLE(df_technic[benchmark], 20)
            del df_technic[benchmark]
            df_technic=self.MaxMinNormalization(df_technic)

            df_y[benchmark + '_pct'] = df_y[benchmark].pct_change(22).shift(-22)
            del df_y[benchmark]
            df_y = self.MaxMinNormalization(df_y)
            for i in range(len(df_sql)):
                query = '''
                SELECT nav_date,close  FROM  {0} where bloomberg_ticker =\'{1}\' and nav_date>'20060101'  order by  nav_date asc
                '''.format(df_sql.iat[i, 3], df_sql.iat[i, 5])
                df_data = pd.read_sql(query, self.conn)
                df_data.rename(columns={"close": df_sql.iat[i, 1]}, inplace=True)
                df_data = self.Date_adj(df_y, df_data)
                df_data = self.Pct_shift(df_data, [df_sql.iat[i, 1]], [df_sql.iat[i, 2]])
                df_data = self.MaxMinNormalization(df_data)
                if i == 0:
                    df_x = df_data.copy()
                else:
                    df_x = pd.merge(df_x, df_data, on='nav_date', how="outer")

            df_ols = pd.merge(df_y, df_x, on='nav_date', how='inner')
            df_ols=pd.merge(df_ols, df_technic,on='nav_date',how='left')
            y = df_ols[benchmark + '_pct']
            x = df_ols[df_sql.X.to_list()+list_technic]
            y.fillna(0, inplace=True)
            x.fillna(0, inplace=True)
            est1 = sm.OLS(y, x)
            est1 = est1.fit()
            est1.summary()
            coef1 = est1.params
            y[len(y) - 1] = np.dot(coef1, x.iloc[-1, :])
            sign = np.round((y.rank(ascending=False) / len(y))[len(y) - 1] * 10, 1)
            df_result = df_result.append([[benchmark, sign]])
        df_result.columns = ['benchmark', 'risk']
        df_benchmark = pd.read_excel(dir + '/Tickers.xlsx', sheet_name='benchmark')
        df_benchmark.rename(columns={"bloomberg_ticker_benchmark": "benchmark"}, inplace=True)
        df_result = pd.merge(df_result, df_benchmark, how='left', on='benchmark')
        df_result.rename(columns={"benchmark":"asset_benchmark","asset":"asset_name"},inplace=True)
        df_result['risk_comment']='no comment'
        df_result['data_source'] = '量化多因子'
        df_result['risk_date']=pd.to_datetime(datetime.date.today())
        df_result=df_result[['risk_date','asset_name','asset_benchmark','risk','data_source','risk_comment']].copy()
        df_sql = df_result[df_result['risk_date'] == df_result['risk_date'].max()].copy()
        df_sql.to_csv('.//'+pd.to_datetime(datetime.date.today()).strftime('%y%m%d')+'.csv',encoding='utf-8-sig',index=False)

        df_sql_riskinfo=df_sql.copy()
        df_sql_riskorigin = df_sql.copy()
        df_sql_riskinfo.rename(columns={"asset_name":"asset","asset_benchmark":"benchmark","risk":"risk_level","risk_comment":"comment"},inplace=True)
        df_sql_riskinfo['is_deleted']=0
        df_sql_riskinfo['create_time']=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        df_sql_riskinfo['modify_time'] = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        return df_result,df_sql_riskorigin,df_sql_riskinfo

    def df_to_sql(self,df,str_name='risk_info',srt_schema='jf_jrm'):
        today_str = datetime.date.today().strftime('%Y%m%d')
        conn = self.conn
        query='''
        delete from {0}.{1} where risk_date>=\'{2}\'and data_source='量化多因子'
        '''.format(srt_schema,str_name,today_str )
        conn.execute(query)
        df.to_sql(name=str_name, schema=srt_schema, con=self.conn, index=False, if_exists='append')

c=multifactor_asset()
df_result,df_sql_riskorigin,df_sql_riskinfo= c.run_ols()
c.df_to_sql( df_sql_riskorigin, str_name='risk_origin', srt_schema='ra_fttw')
c.df_to_sql( df_sql_riskinfo, str_name='risk_info', srt_schema='jf_jrm')
