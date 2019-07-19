import pandas as pd
import datetime

fund = pd.read_excel('nav_col.xlsx', index_col=0)
fund.index = pd.to_datetime(fund.index)
fund1 = fund.iloc[:, [i for i in range(fund.shape[1]) if fund.iloc[:, i].dropna().index.max() >= datetime.datetime(2016, 4, 1)]]
fund1 = fund1.fillna(method='ffill')

fund1.to_excel('nav_col_filled.xlsx')
