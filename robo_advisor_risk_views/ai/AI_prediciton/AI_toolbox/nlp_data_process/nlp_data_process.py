import sys
sys.path.append('..')
import logging
import numpy as np
import pandas as pd
import pickle
from datetime import date
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from Toolbox.data_process.get_news import get_news
from sqlalchemy import create_engine
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')

def data2change(data):
    change = pd.DataFrame(data).pct_change()
    change = change.replace([np.inf, -np.inf], np.nan)
    change = change.fillna(0.).values.tolist()
    change = [c[0] for c in change]
    return change

def news_pivot(data = pd.DataFrame(),news_num=5):
    date_list = list(data.index.unique())
    result = pd.DataFrame(columns=list(range(1,news_num+1)))
    for i in date_list:
        temp = data[data.index==i]
        if temp.shape[0]>=news_num:
            temp = temp.iloc[:news_num,:]
            result.loc[i, :] = temp.T.values
        else:
            result.loc[i,:] = np.append(temp.T.values,[np.nan]*(news_num-temp.shape[0]))
    return result

def load_text_data(con,table_name,benchmark, split_pct,start_date,end_date=date.today(),daily_news_num=5,use_content=False):
    '''
        Load news from csv, group them and split in train/test set due to @date_split
    '''
    df_news = get_news(con=con, news_table=table_name, benchmark=benchmark, start_date=start_date, end_date=end_date,use_content=use_content)

    df_news.index = df_news['report_date']
    df_news = df_news[['title']]
    df_news = news_pivot(data = df_news,news_num = daily_news_num)
    df_news['Combined'] = df_news.iloc[:, :daily_news_num].apply(lambda row: ''.join(str(row.values)), axis=1)

    train_num = int(df_news.shape[0]*split_pct)
    train = df_news.iloc[:train_num, :]
    test = df_news.loc[train_num:, :]

    return train, test

def load_factor_data(con,table_name,asset_name, split_pct,start_date,end_date,predict_len):
    '''
        Load time series from csv, taking adjustment close prices;
        transforming them into percentage of price change;
        split in train/test set due to @date_split
    '''
    sql = "select * from " + table_name + " where bloomberg_ticker = "+ asset_name +" and nav_date between "+ start_date + " and "+ end_date

    data_original = pd.read_sql(sql=sql,con=con)
    train_num = int(data_original.shape[0]*split_pct)
    train = data_original.iloc[:train_num,:]
    test = data_original.iloc[train_num:,:]

    train = train.pct_change(predict_len)
    test = test.pct_change(predict_len)
    return train, test

def text_process(text):
    '''
    Takes in a string of text, then performs the following:
	1. Tokenizes and removes punctuation
	2. Removes  stopwords
	3. Stems
	4. Returns a list of the cleaned text
    '''
    if pd.isnull(text):
        return []

    tokenizer = RegexpTokenizer(r'\w+')
    text_processed = tokenizer.tokenize(text)

    text_processed = [word.lower() for word in text_processed if word.lower() not in stopwords.words('english')]

    porter_stemmer = PorterStemmer()

    text_processed = [porter_stemmer.stem(word) for word in text_processed]

    try:
        text_processed.remove('b')
    except:
        pass

    return " ".join(text_processed)

def transform_text2sentences(train, test, save_train='train_text.p', save_test='test_text.p'):
    '''
        Transforming raw text into sentences,
        if @save_train or @save_test is not None - saves pickles for further use
    '''
    train_text = []
    test_text = []
    for each in train['Combined']:
        train_text.append(text_process(each))
    for each in test['Combined']:
        test_text.append(text_process(each))

    if save_train != None:
        pickle.dump(train_text, open(save_train, 'wb'))
    if save_test != None:
        pickle.dump(test_text, open(save_test, 'wb'))

    return train_text, test_text

def transform_text_into_vectors(train_text, test_text, embedding_size=100, model_path='word2vec10.model'):
    '''
    进行word2vec变化，使每个句子向量化
        Transforms sentences into sequences of word2vec vectors
        Returns train, test set and trained word2vec model
    '''
    data_for_w2v = []
    for text in train_text + test_text:
        words = text.split(' ')
        data_for_w2v.append(words)

    model = Word2Vec(data_for_w2v, size=embedding_size, window=5, min_count=1, workers=4)
    model.save(model_path)
    model = Word2Vec.load(model_path)

    train_text_vectors = [[model[x] for x in sentence.split(' ')] for sentence in train_text]
    test_text_vectors = [[model[x] for x in sentence.split(' ')] for sentence in test_text]

    train_text_vectors = [np.mean(x, axis=0) for x in train_text_vectors]
    test_text_vectors = [np.mean(x, axis=0) for x in test_text_vectors]

    return train_text_vectors, test_text_vectors, model

def split_into_XY(data_chng_train, train_text_vectors, step, window, forecast):
    '''
        Splits textual and time series data into train or test dataset for hybrid model;
        objective y_i is percentage change of price movement for next day
    '''
    X_train, X_train_text, Y_train, Y_train2 = [], [], [], []
    for i in range(0, len(data_chng_train), step):
        try:
            x_i = data_chng_train[i:i + window]
            y_i = np.std(data_chng_train[i:i + window + forecast][3])

            text_average = train_text_vectors[i:i + window]
            last_close = x_i[-1]

            y_i2 = None
            if data_chng_train[i + window + forecast][3] > 0.:
                y_i2 = 1.
            else:
                y_i2 = 0.

        except Exception as e:
            print('KEK', e)
            break

        X_train.append(x_i)
        X_train_text.append(text_average)
        Y_train.append(y_i)
        Y_train2.append(y_i2)

    X_train, X_train_text, Y_train, Y_train2 = np.array(X_train), np.array(X_train_text), np.array(Y_train), np.array(Y_train2)
    return X_train, X_train_text, Y_train, Y_train2
if __name__=="__main__":
    con = create_engine('mysql+pymysql://andrew:andrew@wang@rm-uf679020c6vrt28in7o.mysql.rds.aliyuncs.com:3306/jf_data?charset=utf8')
    table_name = 'news'
    benchmark = 'CL1 Comdty'
    split_pct = 0.8
    start_date='2019-01-12'
    end_date='2019-03-12'
    daily_news_num = 5
    news = load_text_data(con, table_name, benchmark, split_pct, start_date, end_date=date.today(), daily_news_num=daily_news_num, use_content=False)

    pass
