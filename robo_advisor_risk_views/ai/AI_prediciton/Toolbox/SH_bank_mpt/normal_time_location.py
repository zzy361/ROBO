import pandas as pd
import numpy as np
import math
from decimal import Decimal
from datetime import datetime
import os
today = datetime.today()
today = today.strftime('%Y-%m-%d')
origin_sheets = pd.read_excel('2018-03-02.xlsx',[0,1,2,3])

for i in [0,1,2,3]:
    if i in [0, 1]:
        price = pd.read_csv('WechatSH_BANK_ETF_NAV_filled.CSV').fillna(method='ffill')
    elif i in [2, 3]:
        price = pd.read_csv('WechatSH_BANK_ADNAV_filled.CSV').fillna(method='ffill')
    for j in range(len(origin_sheets[i])-1):
        id=origin_sheets[i].iloc[j,2]
        origin_sheets[i].iloc[j,5]=price[id].iloc[-1].item()

    origin_sheets[i]['Value']=origin_sheets[i]['Position']*origin_sheets[i]['NAV']
    origin_sheets[i].iloc[-1, 6]=0
    origin_sheets[i]['Value']=origin_sheets[i]['Value'].map(lambda x:math.floor(x * 100) / 100)
    origin_sheets[i].iloc[-1,6]=sum(origin_sheets[i].iloc[:-1,6])

    origin_sheets[i].iloc[-1,5]=origin_sheets[i].iloc[-1,6]/100000

    origin_sheets[i]['Weight']=origin_sheets[i]['Value']/origin_sheets[i]['Value'].tolist()[-1]
    origin_sheets[i].iloc[:,8]=[0]*len(origin_sheets[i])
    origin_sheets[i].iloc[0,0]=today

print(origin_sheets[3])
