import numpy as np
import pandas as pd


def drift(tgt_pfo, cur_pfo, drift_threshold):
    diff=pd.merge(tgt_pfo,cur_pfo,left_on='id',right_on='asset_id',how='outer')
    diff['diff']=abs(diff.iloc[:,1]-diff.iloc[:,3])
    result = sum(diff['diff']) / 2
    if result >= drift_threshold:
        return 1
    else:
        return 0
