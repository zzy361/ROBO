

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import calendar


def daily_return_ratio(asset, start, end):
    """

    :param asset:
    :param start:
    :param end:
    :return:
    """
    asset_array = np.array(asset.iloc[start:end].as_matrix(columns=None))

    return_ratio = []
    for i in range(0, len(asset.index[start:end]) - 1):
        return_ratio.append(asset_array[i + 1] / asset_array[i] - 1)
    result = pd.DataFrame(return_ratio, index=asset.index[start:end].delete(0), columns=None)
    return result


def standard_deviation(asset, start, end):
    daily_return = daily_return_ratio(asset, start, end)
    ratio = np.array(daily_return.as_matrix(columns=None)).T
    result = ratio.std()
    return result


def find_friday(start_day, end_day):
    friday_list = []
    i = start_day
    while i <= end_day:
        if i.isocalendar()[2] == 5:
            friday_list.append(i)
        i += timedelta(days=1)
    return friday_list


def find_month_end_day(start_day, end_day):
    month_end_day_list = []
    i = start_day
    while i <= end_day:
        if i.day == calendar.monthrange(i.year, i.month)[1]:
            month_end_day_list.append(i)
        i += timedelta(days=1)
    return month_end_day_list


def freq_df_construct(date_list, nav_list, frequency):
    """

    :param date_list:
                        list

    :param nav_list:

    :param frequency:
                        week
    :return:
    """
    nav_df = pd.DataFrame(nav_list, index=date_list, columns=['nav'])
    result = pd.DataFrame(columns=['date_label', 'nav'])
    choosen_index = []

    if frequency == 'day':
        date_label_list = [str(i.year) + 'd' + str(i.timetuple().tm_yday) for i in nav_df.index]
        result['date_label'] = date_label_list
        result['nav'] = nav_list
        result.index = date_list
    elif frequency == 'week':
        friday_list = find_friday(start_day=date_list[0], end_day=date_list[-1])
        for i in friday_list:
            choosen_index.append(nav_df.index[nav_df.index <= i].max())
        choosen_index1 = list(set(choosen_index))
        choosen_index1.sort(key=choosen_index.index)
        date_label_list = [str(i.year) + 'w' + str(i.isocalendar()[1]) for i in choosen_index1]
        new_nav_df = nav_df.loc[choosen_index1, :]
        result['date_label'] = date_label_list
        result['nav'] = list(new_nav_df['nav'])
        result.index = new_nav_df.index
    elif frequency == 'month':
        month_end_day_list = find_month_end_day(start_day=nav_df.index[0], end_day=nav_df.index[-1])
        for i in month_end_day_list:
            choosen_index.append(nav_df.index[nav_df.index <= i].max())
        choosen_index1 = list(set(choosen_index))
        choosen_index1.sort(key=choosen_index.index)
        date_label_list = [str(i.year) + 'm' + str(i.month) for i in choosen_index1]
        new_nav_df = nav_df.loc[choosen_index1, :]
        result['date_label'] = date_label_list
        result['nav'] = list(new_nav_df['nav'])
        result.index = new_nav_df.index
    return result


def volatility_cal(date_list, nav_list, frequency):
    nav_df = freq_df_construct(date_list, nav_list, frequency)
    new_df = nav_df
    del new_df['date_label']
    if frequency == 'day':
        return standard_deviation(new_df, 0, len(new_df)) * np.sqrt(252)
    elif frequency == 'week':
        return standard_deviation(new_df, 0, len(new_df)) * np.sqrt(52)
    if frequency == 'month':
        return standard_deviation(new_df, 0, len(new_df)) * np.sqrt(12)


if __name__ == '__main__':
    year = '10'
    file_name = 'rr3_high'

    file_dir = 'C:/Users/WX/Desktop' + '/portfolio_persent.csv'
    original_df = pd.read_csv(file_dir, encoding='GBK')
    original_df.head()
    date_list = [datetime.strptime(i, "%Y/%m/%d") for i in original_df.iloc[:, 0]]

    a = date_list[100]

    nav_list = list(original_df['total_capital'])
    df = freq_df_construct(date_list, nav_list, 'week')
    a = volatility_cal(date_list, nav_list, 'month')



