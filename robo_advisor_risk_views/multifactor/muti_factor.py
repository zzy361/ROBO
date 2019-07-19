import talib_new as ta

import pandas as pd
import datetime
import numpy as np
import copy
import math
from sqlalchemy import create_engine
import time
import os
import pymysql

aws = 'mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+os.environ['MYSQL_HOST']+':'+os.environ['MYSQL_PORT']

pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
pymysql.converters.conversions = pymysql.converters.encoders.copy()
pymysql.converters.conversions.update(pymysql.converters.decoders)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

s = time.strftime("%Y-%m-%d")

result = pd.DataFrame(columns=['risk_date','asset_name','risk','risk_comment','data_source','asset_benchmark'])
conn = create_engine(aws+'/jf_data?charset=utf8')
raw = pd.read_sql("select * from jf_data.index_ohlcv_pe where nav_date > '2017-01-01' ; ", con=conn)

infos = pd.read_sql("select * from jf_data.index_info ; ", con=conn)

sd = pd.read_excel('asset&index.xlsx')
ss = sd["基准代码"].tolist()
sd.head()
len(ss)

names = sd["大类资产类别"].tolist()
namesdict = dict(zip(ss,names))

ss.remove("LIBMB01M Index")

tem_dict = {'risk_date':s}

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def order_columns(frame, var):
    """
    重新排列的顺序
    :param frame:
    :param var:
    :return:
    """
    var_list = [w for w in frame.columns if w not in var]
    frame = frame[var+var_list]
    return frame

def value_sort(df,column):
    """
    计算因子历史分布，并排序，得到因子权重
    :param df:
    :return:
    """

for it in ss:

    tem_dict["asset_name"] = namesdict[it]

    sample = raw[raw['bloomberg_ticker'] == it].copy(deep=True)
    sample["nav_date"] = pd.to_datetime(sample["nav_date"])
    sample = ta.ATR(sample, 5, ksgn='close')
    sample["std"] = sample["close"].rolling(window=20).std()
    sample["atr_5"] /= sample["atr_5"].max()
    sample["std"] /= sample["std"].max()

    sample = ta.ACCDIST(sample, 5, ksgn='close')

    sample = ta.MA(sample, 5)
    sample = ta.MA(sample, 10)
    sample = ta.MA(sample, 20)
    sample = ta.MA(sample, 30)
    sample = ta.MA(sample, 60)
    sample["at_otput"] = sample["atr_5"].apply(tanh)
    sample["st_otput"] = sample["std"].apply(tanh)
    sample = sample.fillna(0)

    risks = 10*sample["st_otput"].values[-1]
    tem_dict["risk"] = 10 - round(risks,0)
    tem_dict["risk_comment"] = 'no risk_comment'
    tem_dict["data_source"] = '量化多因子'
    tem_dict["asset_benchmark"] = it

    sd = pd.DataFrame.from_dict(tem_dict, orient='index').T
    result = result.append(sd, ignore_index=True)

result = order_columns(result, ['risk_date','asset_name','asset_benchmark','risk','data_source','risk_comment'])
result.head()
result["risk_date"] = pd.to_datetime(result["risk_date"] )

result.to_excel('factor_risk_{0}.xlsx'.format(time.strftime("%Y%m%d")),encoding='gbk',index=False)

print("upload mutil factor sucess")
conns = create_engine(aws+'/ra_fttw?charset=utf8')
result.to_sql(con=conns, name='risk_origin', if_exists='append', index=False)

import json
import requests
loginUrl = 'http://120.27.238.50:8084/QuantSystem/api/v1/user/login'
url = 'http://jrmadmin.thiztech.com/api/insert/riskInfo'

def login(username, password,loginUrl):
    """
    登录
    :param username: 用户名
    :param password: 密码
    :return: 返回0则登录成功,否则返回的是string类型的错误信息
    """
    sess = requests.session()
    sess.headers['Content-Type'] = 'application/json'
    juser = json.dumps({'userAccount': username, 'password': password})
    r = sess.post(loginUrl, data=juser)
    jresponse = json.loads(r.content)

    return jresponse['token'],sess

def write_to_console(df_result):

    token, sess = login(username='pengfei', password='ymlhpf162681', loginUrl=loginUrl)
    dict_json = to_json(df_result, token=token)
    json_to_web(dict_json=dict_json, sess=sess)
    print('sucess')

def json_to_web(dict_json,sess):
    jdata = json.dumps(dict_json)

    r = sess.post(url, data=jdata)
    jresponse = json.loads(r.content)
    if jresponse['errorCode'] != 0:
        return jresponse['errorMsg']
    print('the response is :', jresponse)

def to_json(df_result,token):
    dict_json = {}
    dict_json['columns'] = str(list(df_result.columns)).replace(']','').replace('[','').replace("'",'')
    dict_json["dbName"] =  "jf_jrm"
    dict_json["tableName"]="risk_info"
    dict_json["token"] = token
    df_list = df_result.to_dict(orient='record')
    json_list = [json.dumps(i) for i in df_list]
    dict_json["values"] = df_result.to_dict(orient='record')
    return dict_json
