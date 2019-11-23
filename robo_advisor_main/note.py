# coding=gbk
import pandas as pd
import sqlalchemy
import datetime
import config
import dateutil
import json


def note_pro():
    d = config.g
    d.init()
    db = d.db
    # ��ʼ��Ҫץ�Ķ��ϵ��������

    today = datetime.datetime.today()
    yesterday = (today - dateutil.relativedelta.relativedelta(days=1)).strftime('%Y-%m-%d')
    today = today.strftime('%Y%m%d')

    # ��ץ�������ʲ���Ӧָ����
    risk_map = pd.DataFrame(pd.read_sql('Select asset_benchmark, asset_name, asset_name_english2, FT from ra_fttw.risk_map;', con=db))
    risk_map = risk_map[risk_map['FT'] == 1]
    risk_map_list = str(tuple(risk_map['asset_benchmark'].unique().tolist()))
    risk_date = str(tuple([today, yesterday]))
    # ץ����Ӧ�Ĵ����ʲ�ָ����ʱ��Ķ��ϵ��
    risk_indicator = pd.DataFrame(pd.read_sql('Select risk_date, asset_name, asset_benchmark, risk from ra_fttw.risk_out where asset_benchmark in '+ risk_map_list + 'and risk_date in' + risk_date + ';', con=db))
    risk_indicator = risk_indicator.merge(risk_map, on=['asset_benchmark','asset_name'])
    # �Ѷ��ϵ���������Ҫ�ĸ�ʽ
    def risk_control_pro(indicator, day):
        temp = pd.DataFrame()
        if day == 'today':
            temp = indicator[indicator['risk_date'] == today]
            day = today
        elif day == 'yesterday':
            temp = indicator[indicator['risk_date'] == yesterday]
            day = yesterday
        else: pass
        key = zip(temp['asset_name_english2'], temp['asset_benchmark'])
        value = temp['risk']
        dict_out = {}
        dict_out.update(zip(key, value))
        risk_control_out = {day: dict_out}
        return risk_control_out

    risk_control1 = risk_control_pro(indicator=risk_indicator, day='today')
    risk_control2 = risk_control_pro(indicator=risk_indicator, day='yesterday')

    # ���������ڵĶ��ϵ���ֵ����һ��list��
    risk_control = [risk_control1, risk_control2]

    # ȡ�����׼�¼, ���� last_reg_reb �� last_irreg_reb
    trade_day = (datetime.datetime.today().date() - dateutil.relativedelta.relativedelta(days=120)).strftime('%Y%m%d')
    trade_record = pd.read_sql('select poc_name,asset_ids,trade_date,weight,comment from ra_fttw.trading_record where trade_date>=' + trade_day, con=db)

    trade_last_reg = trade_record[trade_record['comment'] == 'quartly']
    trade_last_reg.sort_values(by='trade_date', ascending=False, inplace=True)
    last_reg_day = trade_last_reg['trade_date'].drop_duplicates(keep='first').tolist()[0]
    last_reg_day = datetime.datetime.strftime(last_reg_day, '%Y%m%d')

    trade_last_irreg = trade_record[trade_record['comment'] != 'quartly']
    trade_last_irreg.sort_values(by='trade_date', ascending=False, inplace=True)
    last_irreg_day = ''
    for i in trade_last_irreg['comment']:
        if not i.isalnum():
            last_irreg_day = trade_last_irreg[trade_last_irreg['comment'] == i]['trade_date'].drop_duplicates(keep='first').tolist()[0]
            last_irreg_day = datetime.datetime.strftime(last_irreg_day, '%Y%m%d')
        else:
            last_irreg_day = None
            pass

    note = str({'last_reg_reb': last_reg_day, 'last_irreg_reb': last_irreg_day, 'risk_control':risk_control})

    return note

if __name__ == '__main__':
    note_out = note_pro()

