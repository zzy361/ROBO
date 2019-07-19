import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def get_data(con,factor_list,table_name,start_date=-1,end_date=1):
    sql=''
    if start_date==-1:
        if type(factor_list) == list:
            factor_list = str(factor_list)
            factor_list = factor_list.replace('[', '(')
            factor_list = factor_list.replace(']', ')')
            sql = 'select * from ' + table_name + ' where bloomberg_ticker in ' + factor_list

        elif factor_list == 'all':
            sql = 'select * from ' + table_name
    else:
        if type(factor_list) == list:
            factor_list = str(factor_list)
            factor_list=factor_list.replace('[','(')
            factor_list=factor_list.replace(']',')')
            sql = 'select * from ' + table_name + ' where bloomberg_ticker in '+ factor_list +' and nav_date between '+'"' + start_date +'"'+ ' and '+'"'+ end_date+'"'

        elif factor_list == 'all':
            sql = 'select * from ' + table_name + ' where nav_date between ' + '"' + start_date + '"' + ' and ' + '"' + end_date + '"'
    factor_data = pd.read_sql(sql=sql,con=con)
    a = factor_data['bloomberg_ticker'].unique()
    print(len(a))
    return factor_data
def macro_data_get():
    pass

def data_process(factor_data = pd.DataFrame(),selected_factor_list=[],back_window=100,corr_threshold=0.5):
    factor_data.fillna(method='ffill',inplace=True)
    factor_data.fillna(method='bfill',inplace=True)
    if len(selected_factor_list) != 0:
        factor_data = factor_data[selected_factor_list]
    factor_data = factor_analysis(factor_data=factor_data,back_window=back_window,corr_threshold=corr_threshold)
    return factor_data

def asset_factor_process(asset_index_data, factor_data = pd.DataFrame(),selected_factor_list=[],back_window=100,corr_num=30):
    factor_data = pd.merge(left=asset_index_data, right=factor_data, left_index=True, right_index=True)
    factor_data.fillna(method='ffill', inplace=True)
    factor_data.fillna(method='bfill', inplace=True)
    if len(selected_factor_list) != 0:
        factor_data = factor_data[selected_factor_list]
    factor_data = asset_factor_colinear_analysis(asset_name=asset_index_data.columns[0],factor_data=factor_data,back_window=back_window,corr_num=corr_num)
    return factor_data

def asset_factor_colinear_analysis(asset_name,factor_data=pd.DataFrame(),back_window=100,corr_num=30):
    factor_data = factor_data.iloc[-back_window:,:]
    factor_data = factor_data.pct_change(1)

    factor_data.fillna(method='ffill',inplace=True)
    factor_data.fillna(method='bfill',inplace=True)
    corr_matrix = factor_data.corr()
    colinear_list = []
    temp = corr_matrix[[asset_name]]
    temp.sort_values(by=asset_name,ascending=False,inplace=True)
    temp = temp.iloc[:corr_num+1]
    temp.drop('close',axis=0,inplace=True)
    temp.dropna(inplace=True)
    temp_list = list(temp.index)
    return temp_list

def factor_analysis(factor_data,back_window,corr_threshold):
    """
    :param factor_data:
    :param back_window:
    :param corr_threshold:
    :return:
    """

    factor_data = colinear_analysis(factor_data=factor_data,back_window=back_window,corr_threshold=corr_threshold)
    return factor_data

def pca_analysis(factor_data, n_pca, using_ratio=True):
    if using_ratio:
        factor_data = factor_data.pct_change(periods=1)
    factor_data.dropna(how='all', inplace=True)
    X = factor_data.loc[:].values
    pca = PCA(n_components=n_pca)
    new_vector = pca.fit_transform(X=X)
    component_matrix = abs(pca.components_)
    arg_max = component_matrix.argmax(axis=1)
    test_list=[]
    representing_factor = np.array(test_list)[arg_max]
    representing_ratio = component_matrix.max(axis=1)
    print(representing_factor)
    return representing_factor, representing_ratio

def cluster_analysis(factor_data, n_clusters):
    """
    聚类算法，注意使用的是大类资产指数对的波动率
    将选择出的高/低相关性宏观经济因子做聚类分析，聚类成9类
    :param test_list:
    :param n_clusters: 聚类后的种类数
    :return: 聚类结果，既每个大类资产对应的聚类类别，0~n_clusters
    """
    factor_data = factor_data.pct_change(periods=1)
    factor_data.dropna(how='all', inplace=True)
    factor_data = factor_data.T
    X = factor_data.loc[:].values
    factor_data["cluster"] = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(X)
    factor_data["ids"] = factor_data.index
    res_id = factor_data["ids"].tolist()
    res_cluster = factor_data["cluster"].tolist()

    return res_cluster

def colinear_analysis(factor_data=pd.DataFrame(),back_window=100,corr_threshold=0.5):
    factor_data = factor_data.iloc[:-back_window,:]
    corr_matrix = factor_data.corr()
    colinear_list = []
    for i in corr_matrix.columns:
        temp = corr_matrix[[i]]
        temp.sort_values(by=i,ascending=False,inplace=True)
        print(temp.shape)
        temp = temp[(temp<corr_threshold)&(temp>-corr_threshold)]
        temp.dropna(inplace=True)
        print(temp.shape)
        temp_list = list(temp.index)
        temp_list.append(i)
        colinear_list.extend(temp_list)
    temp_list = list(np.unique(np.array(colinear_list)))
    result = factor_data[temp_list]
    return result

if __name__=="__main__":
    con = create_engine('mysql+pymysql://andrew:andrew@wang@rm-uf679020c6vrt28in7o.mysql.rds.aliyuncs.com:3306/jf_data?charset=utf8')
    table_name1='index_ohlcv_pe'
    table_name2='eco_nav'
    start_date = '2002-01-01'
    end_date = '2019-01-01'
    factor_list = 'all'

    macro_index_data = pd.read_csv(table_name1 + '.csv')
    macro_index_data = pd.pivot(macro_index_data,index='nav_date',columns='bloomberg_ticker',values='close')
    macro_index_data.index = pd.to_datetime(macro_index_data.index)
    macro_index_data.sort_index(inplace=True)

    macro_factor_data = pd.read_csv(table_name2 + '.csv')
    macro_factor_data = pd.pivot(macro_factor_data, index='nav_date', columns='bloomberg_ticker', values='close')
    macro_factor_data.index = pd.to_datetime(macro_factor_data.index)
    macro_factor_data.sort_index(inplace=True)

    macro_index_data = data_process(factor_data=macro_index_data,corr_threshold=0.02)
    print(macro_index_data.shape)
    macro_factor_data = data_process(factor_data=macro_factor_data,corr_threshold=0.02)
    print(macro_factor_data.shape)
