from log_email import log_email
from config import g
g.init()
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
from config import g
g.init()

@log_email()
def main():
    try:
        # 数据是否更新
        pass
    except:
        pass
    import today_pfo

try:
    main()
    print('successful!!!!!')
except:
    trade_day = (datetime.today().date() - relativedelta(days=120)).strftime('%Y%m%d')
    json = pd.read_sql('select trade_date,json from ra_fttw.trading_record_json where trade_date>=' + trade_day,
                       con=g.db)
    json['trade_date'] = pd.to_datetime(json['trade_date'])
    today_json=json[json['trade_date'] == json['trade_date'].max()]['json'].values[0]
    today_json=eval(today_json)
    for tmp in today_json:
        tmp['data']['data_date']=str(datetime.today().strftime("%Y-%m-%d 00:00:00"))
        tmp['data']['creat_date']=str(datetime.today().strftime("%Y-%m-%d %H:%M:%S"))

    today_str = (datetime.today().date()).strftime('%Y%m%d')
    conn = g.db.connect()
    conn.execute("delete from ra_fttw.trading_record_json where trade_date>=" + today_str)
    conn.close()

    trade_record_json = pd.DataFrame([[datetime.today().date(), str(today_json)]], columns=['trade_date', 'json'])
    trade_record_json.to_sql('trading_record_json', if_exists='append', schema='ra_fttw', con=g.db, index=False)

    print(today_json)

