from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core.expr.expr_pyomo5 import inequality
import pandas as pd
import numpy as np
from weight_1 import weight_1
import os

def mvo(id_list,rtn,vol,rho,percent_list,num_min,num_max,low_b,up_b,asset_cons={'list':[],'indices':[],'lb':[],'ub':[]},cvar_dict={}):
    '''

    :param id_list:
    :param rtn:
    :param vol:
    :param rho:
    :param rtn_exp:
    :param num_min:
    :param num_max:
    :param low_b:
    :param up_b:
    :param asset_cons:
    :return:
    '''
    sigma=pd.DataFrame(np.multiply((np.mat(np.array(vol)**0.5)).T*(np.mat(np.array(vol)**0.5)),rho))
    model = ConcreteModel()
    model.indices = range(0, len(id_list))
    model.w = Var(model.indices, within=Reals)
    model.z = Var(model.indices, domain=Binary)

    model.cons_sum = Constraint(expr=sum([model.w[i]*model.z[i] for i in model.indices])==1)

    model.cons_num=Constraint(expr=inequality(num_min, sum([model.z[i] for i in model.indices]), num_max, strict=False))

    def bound_w(model,i):
        return inequality(low_b,model.w[i],1,strict=False)
    def bound_wz(model,i):
        return inequality(0,model.w[i]*model.z[i],up_b,strict=False)
    model.cons_w=Constraint(model.indices, rule=bound_w)
    model.cons_wz=Constraint(model.indices, rule=bound_wz)
    try:
        def cons_cvar(model):
            return cvar_dict['alpha'] + 1 / (cvar_dict['beta'] * cvar_dict['rtn_matrix'].shape[0]) * sum([model.w[k] * cvar_dict['rtn_matrix'][k] for k in range(cvar_dict['rtn_matrix'].shape[0])]) >= \
                   cvar_dict['expected_cvar']
        model.cons_cvar = Constraint(rule=cons_cvar)
    except:
        pass

    try:
        asset_list=range(0,len(asset_cons['list']))

        def cons_asset(model,j):
            return inequality(asset_cons['lb'][j],sum([model.w[i]*model.z[i] for i in asset_cons['indices'][j]]), asset_cons['ub'][j],
                   strict=False)
        model.cons_asset=Constraint(asset_list,rule=cons_asset)

        def cons_asset1(model,j):
            return inequality(asset_cons['n_min'][j], sum([model.z[i] for i in asset_cons['indices'][j]]), asset_cons['n_max'][j], strict=False)
        model.cons_asset1=Constraint(asset_list,rule=cons_asset1)
    except:
        pass

    opt = SolverFactory('Bonmin', executable=os.path.dirname(os.path.realpath(__file__))+'/bonmin')

    opt.options['tol'] = 1E-2

    model.objective = Objective(expr=sum([model.z[i]*model.z[j]*model.w[i]*model.w[j]*sigma.iloc[i,j] for i in model.indices for j in model.indices]), sense=minimize)
    opt.solve(model)
    min_rtn=sum([model.w[i]._value * model.z[i]._value * rtn[i] for i in model.indices])
    print(min_rtn)
    del model.objective
    model.objective = Objective(expr=sum([model.w[i] * model.z[i] * rtn[i] for i in model.indices]), sense=maximize)
    opt.solve(model)
    max_rtn =sum([model.w[i]._value * model.z[i]._value * rtn[i] for i in model.indices])
    print(max_rtn)
    del model.objective

    results={}
    for k in percent_list:
        print(k)
        model.cons_rtn = Constraint(expr=sum([model.w[i] * model.z[i] * rtn[i] for i in model.indices]) == min_rtn+k*(max_rtn-min_rtn))
        model.objective = Objective(expr=sum([model.z[i]*model.z[j]*model.w[i]*model.w[j]*sigma.iloc[i,j] for i in model.indices for j in model.indices]), sense=minimize)

        opt.solve(model)
        weight_list = []
        for i in model.indices:
            weight_list.append(model.w[i]._value * model.z[i]._value)
        df_w = pd.DataFrame(data=weight_list,index=id_list,columns=['weight'])
        df_w.replace(0,np.nan,inplace=True)
        df_w.dropna(axis=0,inplace=True)
        print(df_w)
        df_w['weight'] = weight_1(df_w['weight'])
        dict_w = df_w.to_dict()['weight']
        del model.objective
        del model.cons_rtn
        results[str(k)]=dict_w
    return results

def yearly_rtn_matrix(nav):
    rtn_matrix = np.array([])
    today = nav.index[-1]
    today = nav.shape[0]

    start_date = 0
    YEAR=365
    year = (today - start_date) / YEAR
    year_list = [year%1]
    if year>1:
        year_list.extend([1]*int(year))
        year_index_list=(np.cumsum(year_list)*YEAR).astype(np.int32)
        year_index_list -=1
        rtn_matrix = ((nav.iloc[year_index_list[0]] / nav.iloc[0]).values) ** (1 / year) - 1
        for i in range(1,len(year_list)):
            rtn_matrix = np.row_stack((rtn_matrix,((nav.iloc[year_index_list[i]] / nav.iloc[year_index_list[i-1]]).values) ** (1 / year) - 1))
    else:
        rtn_matrix = ((nav.iloc[-1] / nav.iloc[0]).values) ** (1 / year) - 1
        rtn_matrix = rtn_matrix[np.newaxis, :]
    return rtn_matrix

def asset_cons_cal(nav,asset_info,bound_info):
    result = {'list': [], 'indices': [], 'lb': [], 'ub': []}
    for i in bound_info.index:
        temp = list(asset_info[asset_info['Category'] == i]['FT_Ticker'])
        indices = [np.argwhere(np.array(list(nav.columns)) == k)[0][0] for k in temp]
        if len(indices) !=0:
            result['list'].append(i)
            result['indices'].append(indices)
            result['lb'].append(0)
            result['ub'].append(bound_info.loc[i,'初始上限'])

    return result
