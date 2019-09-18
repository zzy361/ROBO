import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from config import g
g.init()

def up_bound_cal(asset_list,risk_info,bound_coef=1,expand_coef=1.2):

    asset_bound_level = [1 if i == '货币市场' else 0.4 for i in asset_list]
    risk_level_list = [risk_info.loc[i, 'risk'] for i in asset_list]

    # up_bound=[((risk_level_list[i]/10)**0.7)*asset_bound_level[i] for i in range(0, len(risk_level_list))]
    risk_level = np.array(risk_level_list)
    asset_bound_level = np.array(asset_bound_level) * bound_coef
    row_up_bound = expand_coef * risk_level / sum(risk_level) * np.exp(expand_coef * risk_level / sum(risk_level))
    up_bound = np.sqrt(row_up_bound * asset_bound_level)
    up_bound[up_bound > asset_bound_level] = asset_bound_level[up_bound > asset_bound_level]
    return dict(zip(asset_list, up_bound))
