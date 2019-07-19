import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
import numpy as np
class risk_ruler:
    """
    风控规则类
    """

    def __init__(self):
        self.rule_code = ''
        self.final_risk_signal = 0
        self.poc_name = ''
        self.risk_comment = ''
        self.final_risk_df = pd.DataFrame(columns=['risk_date', 'poc_name', 'risk_signal', 'risk_comment'])

    def big_asset_risk_get(self, con, risk_table_name, trading_record,
                           risk_date, big_asset_list):
        """
        读取对应日期的所有大类资产对应的吉富多空系数
        :param con:
        :param risk_database:
        :param risk_table:
        :param risk_date:
        :return: 返回上次调仓的时候的大类资产的风控信息，以及最近一天的风控信息
        """
        sql = "select * from " + risk_table_name
        original_data = pd.read_sql(sql=sql, con=con)
        former_trading_date = trading_record['trade_date'].values[-1]
        latest_risk_data = original_data[original_data['risk_date'] == original_data['risk_date'].values[-1]]
        latest_risk_data = latest_risk_data[latest_risk_data['asset_name'].isin(big_asset_list)]
        latest_risk_data.index = latest_risk_data['asset_name']
        latest_risk_data = latest_risk_data.reindex(big_asset_list)

        former_day = original_data['risk_date'].unique()[-1]
        former_day_data = original_data[original_data['risk_date'] == former_day]
        former_day_data = former_day_data[former_day_data['asset_name'].isin(big_asset_list)]
        former_day_data.index = former_day_data['asset_name']
        former_day_data = former_day_data.reindex(big_asset_list)
        
        temp_original_data = original_data[original_data['risk_date'] <= former_trading_date]
        former_change_risk_data = temp_original_data[temp_original_data['risk_date'] == temp_original_data['risk_date'].values[-1]]
        former_change_risk_data = former_change_risk_data[former_change_risk_data['asset_name'].isin(big_asset_list)]
        former_change_risk_data.index = former_change_risk_data['asset_name']
        former_change_risk_data = former_change_risk_data.reindex(big_asset_list)

        result_risk = pd.DataFrame()
        result_risk['asset'] = latest_risk_data['asset_name']
        result_risk['former_change_risk'] = former_change_risk_data['risk']
        result_risk['former_day_risk'] = former_day_data['risk']
        result_risk['latest_risk'] = latest_risk_data['risk']
        return result_risk

    def risk_para_get(self, para_con, para_table_name):
        sql = 'select * from ' + para_table_name
        risk_para_df = pd.read_sql(sql=sql, con=para_con, index_col='iid')
        risk_para_dict = risk_para_df[self.__class__.__name__].values[-1]
        risk_para_dict = eval(risk_para_dict)
        return risk_para_dict

    def big_asset_find(self, added_big_asset, fund_info, trading_record):
        """
        找目前持仓对应的大类资产，同时加上人为指定附加的大类资产项
        :param added_big_asset:
        :return:
        """
        latest_trading_record = trading_record[trading_record['trade_date'] == trading_record['trade_date'].values[-1]]
        latest_trading_fund = latest_trading_record['asset_ids'].tolist()
        big_asset_list = fund_info[fund_info['Bloomberg_Ticker'].isin(latest_trading_fund)]['Category'].values
        big_asset_list = np.append(big_asset_list,added_big_asset)
        big_asset_list = list(np.unique(big_asset_list))
        return big_asset_list

    def risk_signal(self, added_big_asset, fund_info, trading_record,
                    con, risk_table_name, final_risk_table_name,
                    para_table_name, risk_date, poc_name):
        """
        总信号
        发出各大类资产风控调仓信号的函数
        :return: 各大类资产风控信号
        """
        self.poc_name = poc_name
        risk_para_dict = self.risk_para_get(para_con=con, para_table_name=para_table_name)

        big_asset_list = self.big_asset_find(added_big_asset, fund_info, trading_record)
        risk_data = self.big_asset_risk_get(con, risk_table_name, trading_record,
                                                                     risk_date, big_asset_list)
        risk_list = []
        risk_list.append(self.diff_signal(risk_data, risk_para_dict['diff_para']))

        risk_list.append(self.minor_risk_signal(risk_data=risk_data, minor_asset_para=risk_para_dict['minor_para']))
        risk_list.append(self.added_asset_diff_risk_signal(risk_data=risk_data, down_para=risk_para_dict['down_para'],added_asset_list=added_big_asset))
        risk_list.append(self.former_day_diff_signal(risk_data=risk_data, down_para=risk_para_dict['former_day_down_para']))

        df_risk = pd.DataFrame(data=risk_list, columns=['risk_source', 'risk_signal'])
        signal_sign = np.max(df_risk['risk_signal'].values)
        today_str = (datetime.today().date()).strftime('%Y%m%d')
        if signal_sign:
            self.risk_comment = self.risk_comment[:-5]
            self.final_risk_signal = 1
            self.final_risk_df.loc[poc_name, :] = [today_str, self.poc_name, self.rule_code, self.risk_comment]
        else:
            self.final_risk_df.loc[poc_name, :] = [today_str, self.poc_name, '0', '']

    def risk_asset_comment_get(self,data=pd.DataFrame()):
        result = '('
        for i in data.index:
            result += str(i)+'='+str(data.loc[i,'latest_risk'])+','
        result = result[:-1]
        return result

    def diff_signal(self, risk_data, down_para):
        """
        信号1
        跟上一次调仓日的信号进行比较，若降得很多，超过阈值，则进行调仓
        :return:
        """
        temp_df = risk_data[risk_data['former_change_risk'].values - risk_data['latest_risk'].values>=down_para]
        if temp_df.shape[0] != 0:
            self.rule_code += '1'
            temp_str = self.risk_asset_comment_get(data=temp_df)
            self.risk_comment += 'diff_signal>=' + str(down_para) + temp_str + '_&_'
            return [self.rule_code, 1]
        else:
            return [self.rule_code, 0]

    def minor_risk_signal(self, risk_data, minor_asset_para):
        """
        信号2
        当其中一个资产的多空系数小于阈值，则发出风控信号
        :return:
        """
        temp_df = risk_data[risk_data['latest_risk'].values <= minor_asset_para]
        if temp_df.shape[0]!=0:
            self.rule_code += '2'
            temp_str = self.risk_asset_comment_get(data=temp_df)
            self.risk_comment += 'minor_risk_signal<=' + str(minor_asset_para) +temp_str+ '_&_'
            return [self.rule_code, 1]
        else:
            self.rule_code += 'minor_risk_signal'
            return [self.rule_code, 0]

    def mdd_risk_signal(self, mdd, mdd_para):
        """
        信号3
        自有信号、最大回撤信号
        此版本不加！！！！！！！！！！！！！！！！！！！！！！
        :return:
        """
        if mdd < mdd_para:
            self.rule_code += '3'
            self.risk_comment += 'mdd_risk_signal <=' + str(mdd_para) + '_&_'
            return [self.rule_code, 1]
        else:
            return [self.rule_code, 0]

    def added_asset_diff_risk_signal(self, risk_data, down_para,added_asset_list):
        """
        信号4
        监控附加大类资产是否产生风控信号,
        :return:
        """
        risk_data = risk_data[risk_data['asset'].isin(added_asset_list)]
        temp_df = risk_data[risk_data['former_change_risk'].values - risk_data['latest_risk'].values >= down_para]
        if temp_df.shape[0]!=0:
            self.rule_code += '4'
            temp_str = self.risk_asset_comment_get(data=temp_df)
            self.risk_comment += 'added_asset_diff_signal>=' + str(down_para) +temp_str+ '_&_'
            return [self.rule_code, 1]
        else:
            return [self.rule_code, 0]

    def former_day_diff_signal(self, risk_data, down_para):
        """
        信号5
        跟前一天的信号进行比较，若降得很多，超过阈值，则进行调仓
        :return:
        """
        temp_df = risk_data[risk_data['former_day_risk'].values - risk_data['latest_risk'].values >= down_para]
        if temp_df.shape[0]!=0:
            self.rule_code += '5'
            temp_str = self.risk_asset_comment_get(data=temp_df)
            self.risk_comment += 'former_day_diff_signal>=' + str(down_para) +temp_str+ '_&_'
            return [self.rule_code, 1]
        else:
            return [self.rule_code, 0]

    def write_to_sql(self, con, table_name):
        today_str = (datetime.today().date()).strftime('%Y%m%d')
        conn = con.connect()
        conn.execute("delete from risk_rules_out where risk_date>=" + today_str)
        conn.close()
        self.final_risk_df.to_sql(name=table_name, con=con,if_exists='append',index=False)
