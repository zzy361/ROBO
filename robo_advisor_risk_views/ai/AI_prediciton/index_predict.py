from AI_Prediction.Lstm.LSTM import *
from sqlalchemy import create_engine
import os
from keras.models import load_model
from sklearn import preprocessing
import pymysql.cursors
import os
import datetime
import pandas as pd
def all_data_table_find(db_connection_dict):
    connection = pymysql.connect(host=db_connection_dict['host'],
                                 user=db_connection_dict['user'],
                                 password=db_connection_dict['password'],
                                 db=db_connection_dict['database_name'],
                                 charset=db_connection_dict['charset'],
                                 cursorclass=pymysql.cursors.DictCursor)
    results = []
    try:
        with connection.cursor() as cursor:
            sql = '''SHOW TABLES'''
            cursor.execute(sql)
            result = cursor.fetchall()
            for i in range(len(result)):
                results.append(result[i]['Tables_in_' + db_connection_dict['database_name']])
    finally:
        connection.close()
    return results


def daily_predict(index_name, history_window, predict_length):
    if os.path.exists(index_name + '_saved_model.h5'):
        model = load_model(index_name + '_saved_model.h5')
        tb = create_engine('mysql://andrew:123456@rm-uf679020c6vrt28in7o.mysql.rds.aliyuncs.com:3306/jfquant_test?charset=utf8')
        db_name = 'table_' + index_name.replace('.', '_')
        sqlasset = "SELECT " + '*' + " FROM " + db_name
        original_data = pd.read_sql(sqlasset, con=tb)
        original_data.dropna(axis=0, inplace=True)
        original_data = original_data[['open', 'high', 'low', 'volume', 'close']]
        new_batch = original_data.iloc[-(history_window+predict_length):, :]# 利用第predict_length天之前history_window的数据组成X
        scaler = preprocessing.StandardScaler()
        batch_scale = scaler.fit(new_batch)
        scaled_data = scaler.transform(new_batch)
        x_new_batch = scaled_data[-(history_window+predict_length):-predict_length, :-1]
        x_new_batch = x_new_batch.reshape(1, x_new_batch.shape[0], x_new_batch.shape[1])
        y_new_batch = np.array([scaled_data[-1, -1]])
        pred = model.predict(x_new_batch)
        real_value_pred = batch_scale.inverse_transform([[0.0, 0.0, 0.0, 0.0, pred]])[0][-1]

        model.fit(
            x_new_batch,
            y_new_batch,
            batch_size=1,
            nb_epoch=1,

            verbose=0)
        model.save(index_name + '_saved_model.h5')
        return real_value_pred
    else:
        tb = create_engine('mysql://andrew:123456@rm-uf679020c6vrt28in7o.mysql.rds.aliyuncs.com:3306/jfquant_test?charset=utf8')
        db_name = 'table_' + index_name.replace('.', '_')
        sqlasset = "SELECT " + '*' + " FROM " + db_name
        original_data = pd.read_sql(sqlasset, con=tb)
        original_data.dropna(axis=0, inplace=True)
        original_data = original_data[['open', 'high', 'low', 'volume', 'close']]
        lstm = Lstm()
        x_train, y_train, x_test, y_test = lstm.preprocess_data(original_data[:: -1], history_window, predict_length=predict_length, split_percent=0.85, problem_class='regression')
        model = lstm.build_model([x_train.shape[2], history_window, 100, 1], units=[200], dropout=0.3, problem_class='regression')
        model.fit(
            x_train,
            y_train,
            batch_size=368,
            nb_epoch=2,
            validation_split=0.1,
            verbose=1)
        model.save('./check_point/'+index_name + '_saved_model.h5')
        return -1


if __name__ == '__main__':
    today = (datetime.datetime.today()-datetime.timedelta(0)).strftime("%Y-%m-%d")
    predict_day = (datetime.datetime.today()+datetime.timedelta(30)).strftime("%Y-%m-%d")
    database_connection = {'host': 'rm-uf679020c6vrt28in7o.mysql.rds.aliyuncs.com', 'user': 'andrew', 'password': '123456', 'database_name': 'jfquant_test', 'charset': 'utf8'}
    all_index = all_data_table_find(database_connection)
    for i in range(len(all_index)):
        all_index[i] = all_index[i].replace('table_', '').replace('_', '.')
    all_index.remove('h11006.csi')
    all_index.remove('h11073.csi')
    all_index.remove('index.info')

    tb = create_engine('mysql+pymysql://andrew:123456@rm-uf679020c6vrt28in7o.mysql.rds.aliyuncs.com:3306/jfquant_test?charset=utf8')
    db_name = 'index_info'
    sqlasset = "SELECT " + '*' + " FROM " + db_name
    index_chinese_name = pd.read_sql(sqlasset, con=tb)
    lower = [i.lower() for i in index_chinese_name['INDEXID_DS']]
    index_chinese_name['INDEXID_DS'] = lower
    index_chinese_name.index = index_chinese_name.iloc[:,0]

    prediction_result = {}
    for i in all_index:
        print(i)
        sqlasset = "SELECT " + '*' + " FROM " + 'table_'+i.replace('.','_')
        index_data = pd.read_sql(sqlasset, con=tb)
        index_data.fillna(method='pad',inplace=True)
        last_close_price = index_data['close'].values[-1]
        prediction_result[i] = round((daily_predict(index_name=i, history_window=100, predict_length=20)-last_close_price)/last_close_price,5)
    index_chinese_name = index_chinese_name.reindex(list(prediction_result.keys()))
    combine_name = index_chinese_name['name']+'_'+index_chinese_name['INDEXID_DS']
    if os.path.exists('prediction_result.xlsx'):
        former_prediction_data = pd.read_excel('prediction_result.xlsx')
        former_prediction_data.loc[predict_day+' VS '+ today] = list(prediction_result.values())
        former_prediction_data.to_excel('prediction_result.xlsx',index=True)
    else:
        df_temp = pd.DataFrame(columns=list(combine_name.values))
        df_temp.to_excel('prediction_result.xlsx')
