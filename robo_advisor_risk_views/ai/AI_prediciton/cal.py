import pandas as pd
from Toolbox.wx_talib import FORCE, BBANDS
original_data = pd.read_csv('000016.SH.csv')
original_data = FORCE(original_data,30)
original_data = BBANDS(original_data, 20)
a=1
