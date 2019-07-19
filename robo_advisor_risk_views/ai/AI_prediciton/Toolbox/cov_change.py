
'''
@author: weixiang
@file: cov_change.py
@time: 2018/4/27 10:37
@desc:

协方差矩阵转置
'''
import numpy as np

def input_data(w,var,ro):
    """
    
    :param w:
    :param var:
    :param ro:
    :return:
    """
    asset_var = 0
    result_sum = 0
    for i in range(len(w)):
        for j in range(len(var)):
            if i == j:
                result_sum += (w[i] ** 2) * (var[i] ** 2)
            else:
                result_sum += (w[i] * w[j]) * np.sqrt(var[i] * var[j])*ro[i, j]
    asset_var = np.sqrt(result_sum / len(w))
    return asset_var



if __name__ == "__main__":
    w = [0.5, 0.5]
    var = [2,2]
    ro = np.array([[1, 0.5],
                   [0.5, 1]])

    re = input_data(w, var ,ro)
    re


