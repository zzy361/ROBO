import sys
import pickle
import os
import glob

a1 = os.path.join(os.getcwd(), "../..")
a2 = os.path.join(os.getcwd(), "..")
a3 = os.path.join(os.getcwd(), ".")
sys.path.append(a1)
sys.path.append(a2)
sys.path.append(a3)

from ai.AI_prediciton.Lstm.LSTM import *
from sqlalchemy import create_engine
from keras.models import load_model
from sklearn import preprocessing
from keras import backend as k
conn = create_engine('mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+os.environ['MYSQL_HOST']+':'+os.environ['MYSQL_PORT']+'/jf_data?charset=utf8')

import pymysql.cursors
import datetime
import pandas as pd
import numpy as np
from ai.AI_prediciton.Toolbox.technical_factor_gneration import feature_calculate
from ai.AI_prediciton.AI_toolbox.dummy_factor_handle import dummy_factor_handle
import warnings
from ai.AI_prediciton.Toolbox.get_factor_data import *
warnings.filterwarnings('ignore')


def get_feature_data(index_name, index_data,macro_data):
    original_data = index_data
    original_data = original_data[['open', 'high', 'low', 'volume', 'close']]
    original_data.columns = ['Open', 'High', 'Low', 'Volume', 'Close']
    feature_data = feature_calculate(df_data=original_data, look_back_days=18, slope_look_back_days=18)
    feature_data.dropna(axis=0, how='any', inplace=True)
    del feature_data['Adj Close']
    feature_data = pd.merge(left=macro_data, right=feature_data,left_index=True,right_index=True)
    dummy_columns = ['Color', 'BarType', 'UpDownVolumeChange']
    feature_data = dummy_factor_handle(feature_data, dummy_columns)
    new_column = list(feature_data.columns)
    new_column.remove('Close')
    new_column.append('Close')
    feature_data = feature_data[new_column]
    return feature_data


def label_assign(feature_data, back_days, class_num):
    if feature_data.shape[1] == 1:
        feature_data.columns = ['Close']
    close_df = feature_data['Close'].to_frame()
    close_rtn = (close_df - close_df.shift(back_days)) / close_df.shift(back_days)
    close_rtn.dropna(axis=0, inplace=True)
    max_rtn = close_rtn.max().values[0]
    min_rtn = close_rtn.min().values[0]
    bins = [min_rtn + i * (max_rtn - min_rtn) / class_num for i in range(class_num)]
    bins.append(max_rtn)
    bins[0] -= 0.01
    df_class = pd.cut(close_rtn['Close'].values, bins, right=True, labels=False)
    if feature_data.shape[1] != 1:
        del feature_data['Close']
    feature_data.drop(feature_data.index[:back_days], inplace=True)

    feature_data['label'] = df_class

    return feature_data


def write_to_sql(df_prediction, big_asset_index_name, df_risk_info, con, database_name):

    conn = con.connect()
    today_str = (datetime.datetime.today().date()).strftime('%Y%m%d')
    conn.execute("delete from risk_origin where risk_date>=" + today_str)
    conn.close()

    df_prediction = df_prediction.T

    df_prediction['benchmark'] = df_prediction.index
    df_prediction = pd.merge(left=df_prediction, right=big_asset_index_name, left_on='benchmark', right_on='asset_benchmark',
                             how='inner')
    df_result = pd.DataFrame(index=list(range(df_prediction.shape[0])), columns=df_risk_info.columns)
    df_result.loc[:, 'iid'] = df_risk_info.index[-1] + np.array(list(range(df_prediction.shape[0]))) + 1
    df_result.loc[:, 'risk_date'] = today_str
    df_result.loc[:, 'asset_name'] = df_prediction['asset_name']
    df_result.loc[:, 'risk'] = df_prediction['risk_level']
    df_result.loc[:, 'risk_comment'] = 'no comment'
    df_result.loc[:, 'data_source'] = 'AI'
    df_result.loc[:, 'asset_benchmark'] = df_prediction['benchmark']

    pd.DataFrame.to_sql(df_result, database_name, con, if_exists='append', index=False)


def daily_predict(index_name, index_data, macro_data,history_window, predict_length, batch_size, epoch, class_num):
    if index_data.shape[1] != 1:
        feature_data = get_feature_data(index_name=index_name, index_data=index_data,macro_data=macro_data)
    else:
        feature_data = pd.merge(left=macro_data, right=index_data, left_index=True, right_index=True)
    feature_data = label_assign(feature_data=feature_data, back_days=predict_length, class_num=class_num)

    feature_data.replace(np.inf, np.nan, inplace=True)
    feature_data.dropna(axis=0, how='any', inplace=True)
    print('feature shape', feature_data.shape)
    print(feature_data.index[-1])
    if os.access('./check_point/' + index_name + '_saved_model.h5', os.F_OK):
        model = load_model('./check_point/' + index_name + '_saved_model.h5')
        new_batch = feature_data.iloc[-(history_window + predict_length):, :-1]
        scaler = preprocessing.StandardScaler()
        batch_scale = scaler.fit(new_batch)
        scaled_data = scaler.transform(new_batch)
        x_new_batch = scaled_data[-(history_window + predict_length):-predict_length, :]
        X = scaled_data[-(history_window):, :]
        X = X.reshape(1, X.shape[0], X.shape[1])
        x_new_batch = x_new_batch.reshape(1, x_new_batch.shape[0], x_new_batch.shape[1])
        Y = np.array([feature_data.iloc[-1, -1]])
        y_new_batch = np.array([0] * class_num)
        y_new_batch[Y] = 1
        y_new_batch = y_new_batch.reshape(1, y_new_batch.shape[0])
        pred = np.argmax(model.predict(X)[0]) + 1



        model.fit(
            x_new_batch,
            y_new_batch,
            batch_size=1,
            nb_epoch=1,

            verbose=0)
        model.save('./check_point/' + index_name + '_saved_model.h5')
        return pred
    else:
        if not os.path.exists('check_point'):
            os.makedirs('check_point')
        lstm = Lstm()
        x_train, y_train, x_test, y_test = lstm.preprocess_data(feature_data, history_window,
                                                                predict_length=predict_length, split_percent=0.85,
                                                                problem_class='multi_classification',
                                                                class_num=class_num)
        model = lstm.build_model([x_train.shape[2], history_window, 100, 1], units=[200], dropout=0.3,
                                 problem_class='multi_classification', class_num=class_num)
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            nb_epoch=epoch,
            validation_split=0.1,
            verbose=1)
        model.save('./check_point/' + index_name + '_saved_model.h5')
        return -1

def big_asset_corresponding_factor_find(asset_index,macro_index_data,macro_factor_data):
    temp_dict = {}
    asset_index = asset_index[['close']]

    macro_index_list = asset_factor_process(asset_index_data=asset_index,factor_data=macro_index_data, corr_num=30)

    macro_factor_list = asset_factor_process(asset_index_data = asset_index,factor_data=macro_factor_data, corr_num=30)
    temp_dict['macro_index'] = macro_index_list
    temp_dict['macro_factor'] = macro_factor_list
    return temp_dict
def all_asset_benchmark_get(con):
    sql = 'select * from risk_map where FT=1'
    big_asset_index_info = pd.read_sql(sql=sql, con=con, index_col='index')
    big_asset_index_info.dropna(axis=0, how='any', inplace=True)
    return big_asset_index_info

def macro_eco_data_get(local=True):
    if local:
        macro_index_data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/macro_index_data.csv',index_col=0)
    else:
        macro_index_data = get_data(con=conn, table_name=table_name1, factor_list=factor_list)
    if local:
        macro_factor_data = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/macro_factor_data.csv', index_col=0)
    else:
        macro_factor_data = get_data(con=conn, table_name=table_name2, factor_list=factor_list)
    macro_index_data = pd.pivot(macro_index_data, index='nav_date', columns='bloomberg_ticker', values='close')
    macro_index_data.index = pd.to_datetime(macro_index_data.index)
    macro_index_data.sort_index(inplace=True)

    macro_factor_data = pd.pivot(macro_factor_data, index='nav_date', columns='bloomberg_ticker', values='close')
    macro_factor_data.index = pd.to_datetime(macro_factor_data.index)
    macro_factor_data.sort_index(inplace=True)

    macro_index_data.fillna(method='ffill', inplace=True)
    macro_index_data.fillna(method='bfill', inplace=True)

    macro_factor_data.fillna(method='ffill', inplace=True)
    macro_factor_data.fillna(method='bfill', inplace=True)

    return macro_index_data, macro_factor_data


def regular_factor_data_get():
    pass




today = (datetime.datetime.today() - datetime.timedelta(0)).strftime('%Y-%m-%d %H:%M:%S')
predict_day = (datetime.datetime.today() + datetime.timedelta(30)).strftime("%Y-%m-%d")

conn = create_engine('mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+os.environ['MYSQL_HOST']+':'+os.environ['MYSQL_PORT']+'/jf_data?charset=utf8')
conn1 = create_engine('mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+os.environ['MYSQL_HOST']+':'+os.environ['MYSQL_PORT']+'/jf_jrm?charset=utf8')
conn_ra = create_engine('mysql+pymysql://'+os.environ['MYSQL_USER']+':'+os.environ['MYSQL_PASSWORD']+'@'+os.environ['MYSQL_HOST']+':'+os.environ['MYSQL_PORT']+'/ra_fttw?charset=utf8')


db_name = 'index_ohlcv_pe'

big_asset_index_info = all_asset_benchmark_get(con=conn_ra)

index_chinese_name = list(big_asset_index_info['asset_name'])
all_index = list(big_asset_index_info['asset_benchmark'])
all_index_str = str(all_index)
all_index_str = all_index_str.replace('[', '(')
all_index_str = all_index_str.replace(']', ')')
sqlasset = "SELECT " + '*' + " FROM " + db_name + ' WHERE bloomberg_ticker in ' + all_index_str
available_column = ['bloomberg_ticker', 'nav_date', 'open', 'high', 'low', 'close', 'volume']
original_data = pd.read_sql(sqlasset, con=conn)

original_data = original_data[available_column]

original_data.replace(0, np.nan, inplace=True)

original_data.index = original_data['nav_date']
original_data.index = pd.to_datetime(original_data.index)
original_data.sort_index(inplace=True)

total_index_list = list(set(original_data['bloomberg_ticker']))

db_name = 'risk_info'
db_name_ra = 'risk_origin'
sqlasset1 = "SELECT " + '*' + " FROM " + db_name_ra
df_risk_info = pd.read_sql(sqlasset1, con=conn_ra)

batch_size = 100
epoch = 3
prediction_result = {}
table_name1='index_ohlcv_pe'
table_name2='eco_nav'
factor_list='all'
class_num = 10



macro_index_data_daily,macro_factor_data_daily = macro_eco_data_get(local=False)

with open(os.path.dirname(os.path.realpath(__file__)) + "/asset_macro_factor.file", "rb") as f:
    asset_macro_factor = pickle.load(f)
model_num = len(glob.glob(pathname='./check_point/'))
print('the model number in the check_point file is ======= ', model_num)
if os.path.exists('check_point'):
    for i in os.listdir('check_point'):
        print(i,'\n')
else:
    print('the check_point file does not exist!!!!!!')

for i in all_index: 
    print(i)
    if i in total_index_list and os.access('./check_point/' + i + '_saved_model.h5', os.F_OK):
        index_data = original_data[original_data['bloomberg_ticker'] == i]
        del index_data['bloomberg_ticker']
        index_data.index = index_data['nav_date']
        index_data.index = pd.to_datetime(index_data.index)
        del index_data['nav_date']
        close = index_data['close'].to_frame()
        close.dropna(axis=0, how='any', inplace=True)
        index_data.dropna(axis=0, how='any', inplace=True)

        if index_data.shape[0] < 1:
            index_data = close
            index_data.columns=['Close']
        macro_index_data_temp = macro_index_data_daily[asset_macro_factor[i]['macro_index']]

        macro_factor_data_temp= macro_factor_data_daily[asset_macro_factor[i]['macro_factor']]
        macro_data_temp = pd.merge(left=macro_index_data_temp, right=macro_factor_data_temp, how='outer', left_index=True,right_index=True)
        macro_data_temp.fillna(method='ffill', inplace=True)
        macro_data_temp.fillna(method='bfill', inplace=True)
        print(index_data.shape)
        print(macro_index_data_temp.shape)
        print(macro_factor_data_temp.shape)
        print(macro_data_temp.shape)

        predict_value = daily_predict(index_name=i, index_data=index_data, macro_data=macro_data_temp,history_window=100, predict_length=20,
                                      batch_size=batch_size, epoch=epoch, class_num=class_num)
        prediction_result[i] = predict_value
    else:
        continue

df_prediction = pd.DataFrame(prediction_result, index=['risk_level'])

if -1 not in prediction_result.values():
    write_to_sql(df_prediction=df_prediction, big_asset_index_name=big_asset_index_info, df_risk_info=df_risk_info, con=conn_ra, database_name=db_name_ra)
k.clear_session()
