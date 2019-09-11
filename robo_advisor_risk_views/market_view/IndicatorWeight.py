import sys
import time

import datetime
import logging
import numpy as np
import os
import pandas as pd
from sqlalchemy import create_engine

aws = 'mysql+pymysql://' + os.environ['MYSQL_USER'] + ':' + os.environ['MYSQL_PASSWORD'] + '@' + os.environ['MYSQL_HOST'] + ':' + os.environ['MYSQL_PORT']
conn = aws


class LongShort_Indicator():
    def __init__(self, dict_input):
        self.conn = create_engine(conn)
        self.alpha = dict_input['alpha']
        self.pomax = dict_input['pomax']
        self.pomin = dict_input['pomin']
        self.npow = dict_input['npow']
        query = '''
        SELECT distinct( nav_date) FROM jf_data.index_ohlcv_pe where bloomberg_ticker in("SHSN300 Index","SPX Index") order by nav_date asc
        '''
        data_TradingDay = pd.read_sql(query, self.conn)
        data_TradingDay.loc[len(data_TradingDay), 'nav_date'] = pd.to_datetime(datetime.date.today())
        data_TradingDay.drop_duplicates('nav_date', keep='first', inplace=True)
        data_TradingDay['nav_date_1month'] = data_TradingDay['nav_date'].shift(-22).dropna(0)
        self.nav_date = data_TradingDay.copy()
        data_TradingDay = data_TradingDay[pd.notnull(data_TradingDay['nav_date_1month'])].copy()

        self.data_TradingDay = data_TradingDay

    def get_pct_month(self, data, day):
        data['Pct'] = data['close'].pct_change(periods=day, fill_method='pad')
        data.dropna(inplace=True)
        return data

    def Date_adj(self, data2):
        date_nav = self.nav_date
        data2 = pd.merge(date_nav, data2, on='nav_date', how='outer')
        data2['Comb'] = data2['Strategy'] + "*" + data2["benchmark"]
        data2.drop_duplicates(['Comb', 'nav_date'], inplace=True)
        pvt_predict = data2.pivot(columns='Comb', index='nav_date', values='Sign')
        del pvt_predict[pvt_predict.columns[0]]
        for j in range(len(pvt_predict.columns)):
            for i in range(1, len(pvt_predict)):
                if (pvt_predict.iat[i, j]) >= 0:
                    pvt_predict.iat[i, j] = pvt_predict.iat[i, j]
                else:
                    pvt_predict.iat[i, j] = pvt_predict.iat[i - 1, j]

        temp = pd.DataFrame(pvt_predict.stack())
        temp.columns = ['Sign']
        temp.reset_index(inplace=True)
        temp['benchmark'] = list(map(lambda x: x.split('*')[1], temp['Comb']))
        temp['Strategy'] = list(map(lambda x: x.split('*')[0], temp['Comb']))
        return temp[["nav_date", "benchmark", "Strategy", "Sign"]]

    def map_pencentile(self, dat2, dat3):
        data2 = dat2.copy()
        data3 = dat3.copy()
        list_colname = data2.columns.to_list()
        for i in range(len(data2)):
            for j in range(1, len(data2.columns)):
                colname = list_colname[j]
                for k in range(1, len(data3)):
                    if (data2.iat[i, j] <= data3.loc[k, colname]) and (data2.iat[i, j] > data3.loc[k - 1, colname]):
                        data2.iat[i, j] = data3.loc[k, 'index'] / 10.0
                        break
                    elif data2.iat[i, j] > data3.loc[98, colname]:
                        data2.iat[i, j] = 10
                    elif data2.iat[i, j] <= data3.loc[0, colname]:
                        data2.iat[i, j] = data3.loc[0, 'index'] / 10.0
        data_realpercentile = data2.copy()
        data_realpercentile.set_index('nav_date', inplace=True)
        data_realpercentile = data_realpercentile.unstack()
        data_realpercentile = pd.DataFrame(data_realpercentile)
        data_realpercentile.reset_index(inplace=True)
        data_realpercentile.rename(columns={"level_0": "benchmark", 0: "Indicator_real"}, inplace=True)

        return data_realpercentile

    def get_distance(self, sub_data):
        for i in range(2, len(sub_data)):
            if ~np.isnan(sub_data.iloc[i - 1, 'Distance']) and ~np.isnan(sub_data.iloc[i, 'Distance']):
                sub_data.iloc[i, 'Distance'] = self.alpha * sub_data.iloc[i, 'Distance'] + (1 - self.alpha) * \
                                               sub_data.iloc[i - 1, 'Distance']
            elif ~np.isnan(sub_data.iloc[i - 1, 'Distance']) and np.isnan(sub_data.iloc[i, 'Distance']):
                sub_data.iloc[i, 'Distance'] = sub_data.iloc[i - 1, 'Distance']
            elif np.isnan(sub_data.iloc[i - 1, 'Distance']) and ~np.isnan(sub_data.iloc[i, 'Distance']):
                sub_data.iloc[i, 'Distance'] = sub_data.iloc[i - 1, 'Distance']
            else:
                sub_data.iloc[i, 'Distance'] = np.nan
        return sub_data

    def get_oppsweight(self, data1, data2):

        Alphaconst = self.alpha
        Data = pd.merge(data1, data2, how='left', on=['nav_date', 'benchmark'])
        Data["Distance"] = (Data["Indicator_real"] - Data["Sign"]).abs()
        Data['BS'] = Data['benchmark'] + '*' + Data['Strategy']
        pvt_dis = Data.pivot(index='nav_date', columns='BS', values='Distance')
        df_D = pvt_dis.copy()
        for j in range(len(df_D.columns)):
            for i in range(1, len(df_D)):
                if ~np.isnan(df_D.iloc[i - 1, j]) and ~np.isnan(df_D.iloc[i, j]):
                    df_D.iloc[i, j] = Alphaconst * df_D.iloc[i, j] + (1 - Alphaconst) * \
                                      df_D.iloc[i - 1, j]
                elif ~np.isnan(df_D.iloc[i - 1, j]) and np.isnan(df_D.iloc[i, j]):
                    df_D.iloc[i, j] = df_D.iloc[i - 1, j]
                elif np.isnan(df_D.iloc[i - 1, j]) and ~np.isnan(df_D.iloc[i, j]):
                    df_D.iloc[i, j] = df_D.iloc[i, j]
                else:
                    df_D.iloc[i, j] = np.nan
        df_exp = -df_D.copy()
        for i in range(len(df_exp)):
            for j in range(len(df_exp.columns)):
                df_exp.iat[i, j] = pow(self.npow, df_exp.iat[i, j])
        df_exp = df_exp.apply(np.exp)
        df_exp = pd.DataFrame(df_exp.stack())
        df_exp.columns = ['Distance_exp']
        df_exp.reset_index(inplace=True)
        df_exp['benchmark'] = list(map(lambda x: x.split("*")[0], df_exp['BS']))
        df_exp['Strategy'] = list(map(lambda x: x.split("*")[1], df_exp['BS']))
        df_sum = pd.DataFrame(df_exp.groupby(["nav_date", "benchmark"])["Distance_exp"].apply(np.sum))
        df_sum.reset_index(inplace=True)
        df_sum.rename(columns={"Distance_exp": "Distance_sum"}, inplace=True)
        df_exp = pd.merge(df_exp, df_sum, on=['nav_date', 'benchmark'], how='left')
        df_exp['Weight'] = df_exp['Distance_exp'] / df_exp['Distance_sum']
        df_exp = pd.merge(df_exp, self.data_TradingDay, on=['nav_date'], how='inner')
        result_weight = df_exp[["nav_date_1month", "benchmark", "Strategy", "Distance_exp", "Weight"]].copy()
        result_weight.rename(columns={"nav_date_1month": "nav_date", "Distance_exp": "Distance_wema"}, inplace=True)
        result_weight = pd.merge(result_weight, data1[["nav_date", "Sign", "benchmark", "Strategy"]], how='left', on=["nav_date", "benchmark", "Strategy"])
        return result_weight

    def risk_polimit(self, result_weight):
        result_weight["Weight"] = np.where(result_weight["Weight"] > self.pomax, self.pomax, np.where(result_weight["Weight"] < self.pomin, self.pomin, result_weight["Weight"]))
        df_sum = pd.DataFrame(result_weight.groupby(["nav_date", "benchmark"])["Weight"].apply(np.sum))
        df_sum.reset_index(inplace=True)
        df_sum.rename(columns={"Weight": "Weight_sum"}, inplace=True)
        result_weight = pd.merge(result_weight, df_sum, on=['nav_date', 'benchmark'], how='left')
        result_weight["Weight"] = result_weight["Weight"] / result_weight["Weight_sum"]
        return result_weight

    def get_oppsrisk_level(self, result_weight):

        result_weight["risk_mix"] = result_weight["Sign"] * result_weight["Weight"]
        result_risk_mix = pd.DataFrame(result_weight.groupby(by=["nav_date", "benchmark"])["risk_mix"].apply(np.sum))
        result_risk_mix.reset_index(inplace=True)

        result_risk_mix["risk_mix"] = list(map(lambda x: round(x, 1), result_risk_mix["risk_mix"]))
        result_risk_mix['Strategy'] = "吉富多空系数"
        result_risk_mix.rename(columns={"nav_date": "risk_date", "benchmark": "asset_benchmark", "risk_mix": "risk", "Strategy": "data_source"}, inplace=True)

        result_risk_mix = pd.merge(result_risk_mix, self.data_asset, on=['asset_benchmark'], how='left')
        return result_risk_mix[['risk_date', 'asset_name', 'asset_benchmark', 'risk', 'data_source']]

    def get_data_percentile(self):
        query_percent = '''SELECT * FROM ra_fttw.assetquantile_1month'''
        data_percentile = pd.read_sql(query_percent, self.conn)

        return data_percentile

    def get_data_realandpre(self, data_percentile):

        query_pre = '''SELECT * FROM ra_fttw.risk_origin'''
        df_pre = pd.read_sql(query_pre, self.conn)
        query_map = '''SELECT * FROM ra_fttw.risk_map '''
        data_asset = pd.read_sql(query_map, self.conn)
        data_asset = data_asset[["asset_name", "asset_benchmark"]].copy()

        self.data_asset = data_asset
        data_predict = df_pre[['risk_date', 'risk', 'asset_benchmark', 'data_source']].copy()
        data_predict.columns = ['nav_date', 'Sign', 'benchmark', 'Strategy']
        data_predict = data_predict[data_predict['benchmark'].isin(data_percentile.columns.to_list())].copy()
        data_predict['nav_date'] = list(map(lambda x: x.strftime('%Y-%m-%d'), data_predict['nav_date']))
        data_predict['nav_date'] = pd.to_datetime(data_predict['nav_date'])
        list_benchmark = data_predict["benchmark"].unique().tolist()
        data_real = data_predict[["nav_date", "benchmark"]].copy()
        for i in range(len(list_benchmark)):
            query = '''
            SELECT nav_date,close FROM jf_data.index_ohlcv_pe  where bloomberg_ticker=\'{0}\' and nav_date>=20091231 order by nav_date asc
            '''.format(list_benchmark[i])
            df = pd.read_sql(query, self.conn)
            df = self.get_pct_month(df, 22)
            df['Pct'] = df['Pct'].shift(-22)
            df.rename(columns={"Pct": list_benchmark[i]}, inplace=True)
            data_real = pd.merge(data_real, df[["nav_date", list_benchmark[i]]], how='outer', on="nav_date")
            data_real.sort_values("nav_date", ascending=True, inplace=True);
        data_real.fillna(method='ffill', inplace=True)
        del data_real["benchmark"]
        data_real.drop_duplicates('nav_date', keep='first', inplace=True);
        data_real.reset_index(inplace=True, drop=True)
        data_real = data_real.iloc[0:len(data_real) - 24].copy()
        data_predict.drop_duplicates(["nav_date", "benchmark", "Strategy"], keep='last', inplace=True)
        data_predict = self.Date_adj(data_predict)
        return data_real, data_predict

    def run_version1(self, list_S):
        data_percentile = self.get_data_percentile()
        data_real, data_predict = self.get_data_realandpre(data_percentile)
        data_predict = data_predict[data_predict['Strategy'].isin(list_Strategy)].copy(0)
        data_realpercentile = self.map_pencentile(data_real, data_percentile)
        result_weight = self.get_oppsweight(data_predict, data_realpercentile)
        result_weight = self.risk_polimit(result_weight)
        result_risk_mix = self.get_oppsrisk_level(result_weight)
        return result_weight, result_risk_mix

    def run_versionaws(self, list_Strategy):
        data_percentile = self.get_data_percentile()
        data_real, data_predict = self.get_data_realandpre(data_percentile)
        data_predict = data_predict[data_predict['Strategy'].isin(list_Strategy)].copy(0)
        data_realpercentile = self.map_pencentile(data_real, data_percentile)
        result_weight = self.get_oppsweight(data_predict, data_realpercentile)
        result_weight = self.risk_polimit(result_weight)
        result_risk_mix = self.get_oppsrisk_level(result_weight)
        df_weight = result_weight[['nav_date', 'benchmark', 'Strategy', 'Weight']].copy()
        df_weight['Comb'] = list(map(lambda x, y: str(x) + '_' + y, df_weight['nav_date'], df_weight['benchmark']))
        pvt_df1 = df_weight.pivot(columns='Strategy', index='Comb', values='Weight')

        for i in range(len(list_Strategy)):
            if i == 0:
                pvt_df1['risk_comment'] = list(map(lambda x: list_Strategy[i] + '_' + str(round(x, 2)), pvt_df1[list_Strategy[i]]))
            else:
                pvt_df1['risk_comment'] = list(map(lambda x, y: x + '__' + list_Strategy[i] + '_' + str(round(y, 2)), pvt_df1['risk_comment'], pvt_df1[list_Strategy[i]]))
        pvt_df1.reset_index(inplace=True)
        pvt_df1['risk_date'] = list(map(lambda x: x.split('_')[0], pvt_df1['Comb']))
        pvt_df1['asset_benchmark'] = list(map(lambda x: x.split('_')[1], pvt_df1['Comb']))
        pvt_df1['risk_date'] = pd.to_datetime(pvt_df1['risk_date'])
        df_result = pd.merge(result_risk_mix, pvt_df1[['risk_date', 'asset_benchmark', 'risk_comment']], on=['asset_benchmark', 'risk_date'], how='left')
        df_result.dropna(inplace=True)
        df_sql = df_result[df_result['risk_date'] == df_result['risk_date'].max()].copy()
        df_riskout = df_sql.copy()
        df_riskinfo = df_sql.copy()
        df_riskinfo.rename(columns={"asset_name": "asset", "asset_benchmark": "benchmark", "risk": "risk_level",
                                    "risk_comment": "comment"}, inplace=True)
        df_riskinfo['is_deleted'] = 0
        df_riskinfo['create_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        df_riskinfo['modify_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        return df_result, df_riskout, df_riskinfo

    def df_to_sql(self, df, str_name='risk_out', srt_schema='ra_fttw'):
        today_str = datetime.date.today().strftime('%Y%m%d')
        conn = self.conn
        query = '''
        delete from {0}.{1} where risk_date>=\'{2}\'and data_source='吉富多空系数'
        '''.format(srt_schema, str_name, today_str)
        conn.execute(query)
        df.to_sql(name=str_name, schema=srt_schema, con=self.conn, index=False, if_exists='append')


list_Strategy = ['AI', '量化多因子', '吉富投资']  # $['AI', '量化多因子', '吉富投资', '吉富多空系数', 'AI预测算法']
dict_input = {"alpha": 0.25, "pomax": 1, "pomin": 0.0, 'npow': 2.5}
LSIndicator = LongShort_Indicator(dict_input)
df_result, df_riskout, df_riskinfo = LSIndicator.run_versionaws(list_Strategy)
LSIndicator.df_to_sql(df=df_riskout, str_name='risk_out', srt_schema='ra_fttw')
LSIndicator.df_to_sql(df=df_riskinfo, str_name='risk_info', srt_schema='jf_jrm')
