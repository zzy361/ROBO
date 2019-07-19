import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from config import g
g.init()

def up_bound_cal(asset_list,risk_info):

    asset_bound_level = [1 if i == '货币市场' else 0.4 for i in asset_list]
    risk_level_list = [risk_info.loc[i, 'risk'] for i in asset_list]

    up_bound=[((risk_level_list[i]/10)**0.7)*asset_bound_level[i] for i in range(0,len(risk_level_list))]

    return dict(zip(asset_list, up_bound))
