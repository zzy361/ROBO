
import pandas as pd
import matplotlib.pyplot as plt
def factor_correlation_cal(df,label_column,back_window):
    df = df.iloc[-back_window:,:]
    corr_df = df.corr()
    corr_df.drop(label_column,axis=0,inplace=True)
    return corr_df.loc[:,label_column]

if __name__=='__main__':
    df = pd.read_csv('df.csv',index_col=0)
    df.dropna(inplace=True)
    result=factor_correlation_cal(df,'Close',100)
    result.plot(kind='bar')
    plt.show()
