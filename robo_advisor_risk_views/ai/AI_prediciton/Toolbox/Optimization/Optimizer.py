import importlib
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
from Toolbox.Optimization.PSO.PSO import PSO
import numpy as np


class Optimizer:
    def __init__(self, optimizer_para_dict, backtest_para_dict, opt_para_dict, draw_opt_line, bounds):
        self.optimizer_para_dict = optimizer_para_dict
        self.backtest_para_dict = backtest_para_dict
        self.draw_opt_line = draw_opt_line
        self.opt_para_dict = opt_para_dict
        self.bounds = bounds

    def calculate(self):
        self.optimizer_para_dict['opt_algorithm'] = self.optimizer_para_dict['opt_algorithm'].upper()
        module_dir = 'Toolbox.Optimization.' + self.optimizer_para_dict['opt_algorithm'] + '.' + self.optimizer_para_dict['opt_algorithm']
        opt = importlib.import_module(module_dir)
        optimizing_algorithm = getattr(opt, self.optimizer_para_dict['opt_algorithm'])(optimizer_para_dict=self.optimizer_para_dict,
                                                                                       backtest_para_dict=self.backtest_para_dict,
                                                                                       bound=self.bounds,
                                                                                       opt_para_dict=self.opt_para_dict
                                                                                       )
        best_para = optimizing_algorithm.solve(self.draw_opt_line)
        return best_para


if __name__ == '__main__':
    opt_algorithm = 'pso'
    para_dict = {}
    if opt_algorithm == 'pso':
        para_dict = {'w': 0.8,
                     'c1': 2,
                     'c2': 2,
                     'r1': 0.6,
                     'r2': 0.3}
        optimize_para_dict = {'para_dict': para_dict,
                              'opt_algorithm': 'pso',
                              'pop_size': 3,
                              'iter_num': 3
                              }
    elif opt_algorithm == 'ga':
        pass

    backtest_para_dict = {
        'start_date': '20150101',
        'end_date': '20170101',
        'percentile': 0.5,
    }
    back_para = {
        'black_window': [5, 30],
        'tolerance': [0.03, 0.15],
        "down_risk": [7, 11],
        "risk_off": [4, 7],
        "downside_window": [20, 60],
        "risk_weight": [0, 0.6]
    }
    bounds = np.array(list(back_para.values())).T
    draw_opt_line = 1
    obj = Optimizer(optimizer_para_dict=optimize_para_dict, backtest_para_dict=backtest_para_dict, opt_para_dict=back_para,
                    draw_opt_line=draw_opt_line, bounds=bounds)
    best_para = obj.calculate()
    print("The optimal parameters are:", best_para)
