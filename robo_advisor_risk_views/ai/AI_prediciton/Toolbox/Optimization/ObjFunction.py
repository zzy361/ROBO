
from Backtest_mars3.Backtest1 import back_test
from para_function import *


def obj_function(chrom, opt_para_dict, backtest_para_dict):
    for i in range(len(opt_para_dict.keys())):
        opt_para_dict[list(opt_para_dict.keys())[i]] = chrom[i]
    nav = back_test(opt_para_dict,backtest_para_dict)
    std = standard_deviation(nav, nav.index[0], nav.index[-1])
    annual_rtn = annualized_return_ratio(nav, nav.index[0], nav.index[-1])
    sharp = annual_rtn / std
    a = max_draw_down(nav, nav.index[0], nav.index[-1])
    mdd = max_draw_down(nav, nav.index[0], nav.index[-1])

    score = annual_rtn
    return score
