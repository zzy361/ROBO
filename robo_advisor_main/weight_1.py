from decimal import Decimal

def weight_1(weight_series):
    """
    对权重的小数点进行截取，并将权重之和为1
    :param weight_series: 权重对应的series
    :return: 修正后的权重对应的series
    """
    weight_series = weight_series.fillna(0)
    minidx = weight_series[weight_series > 0].idxmin()
    maxidx = weight_series.idxmax()
    weight_series = weight_series.apply(lambda x: Decimal(x).quantize(Decimal('0.00')))
    if weight_series.sum() < 1:
        weight_series[minidx] += 1 - weight_series.sum()
    elif weight_series.sum() > 1:
        weight_series[maxidx] += 1 - weight_series.sum()

    return weight_series.apply(lambda x: float(x))
