import pandas as pd
import datetime


fund=pd.read_csv('/Users/mars/JIFU/python project/JFquant/Backtest/data/SH_BANK_ETF_NAV.CSV',index_col=0)
fund.index=pd.to_datetime(fund.index)
fund1= fund.iloc[:, [i for i in range(fund.shape[1]) if fund.iloc[:, i].dropna().index.max() >= datetime.datetime(2018, 2, 1)]]
fund1=fund1.fillna(method='ffill')
fund1.to_csv('/Users/mars/JIFU/python project/JFquant/Backtest/data/SH_BANK_ETF_NAV_filled.CSV')
