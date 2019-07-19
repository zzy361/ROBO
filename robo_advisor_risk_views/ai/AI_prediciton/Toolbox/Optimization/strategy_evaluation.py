def strategy_evaluation(strategy_sharpe_ratio, strategy_mdd, strategy_std, strategy_return):
    weight_dict = {'sharpe_ratio': 0.4, 'mdd': 0.4, 'std': 0.1, 'return': 0.1}
    result = weight_dict['sharpe_ratio'] * strategy_sharpe_ratio - weight_dict['mdd'] * strategy_mdd - weight_dict['std'] * strategy_std + weight_dict['return'] * strategy_return
    return result
