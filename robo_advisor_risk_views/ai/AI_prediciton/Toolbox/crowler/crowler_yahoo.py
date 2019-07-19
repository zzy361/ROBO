import urllib.request as urllib2
import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
from datetime import datetime
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys
def get_historical_data(name, start_day,end_day):
    start_int = int(datetime.strptime(start_day,'%Y-%m-%d').timestamp())
    end_int = int(datetime.strptime(end_day, '%Y-%m-%d').timestamp())
    url = "https://finance.yahoo.com/quote/" + name + "/history/"+"?period1="+str(start_int)+"&period2="+str(end_int)+"&interval=&interval=1d&filter=history&frequency=1d"
    option = webdriver.ChromeOptions()
    option.add_argument("headless")
    web = webdriver.Chrome('E:/Program Files/chromedriver.exe',chrome_options=option)

    js = "window.scrollTo(0,100000)"

    web.get(url=url)
    print(len(web.page_source))
    org=0
    for i in range(5):
        web.execute_script(js)
        print(len(web.page_source))
        time.sleep(3)
        a=web.execute_script("return document.querySelector('#render-target-default').clientHeight")
        org=a


    rows = bs(web.page_source).findAll('table')[0].tbody.findAll('tr')
    result=pd.DataFrame(columns=['open','high','low','close','adj_close','volume'])
    for each_row in rows:
        divs = each_row.findAll('td')

        result.loc[divs[0].span.text.replace(',', '')] = [float(divs[1].span.text.replace(',', '')),float(divs[2].span.text.replace(',', '')),float(divs[3].span.text.replace(',', '')),float(divs[4].span.text.replace(',', '')),float(divs[5].span.text.replace(',', '')),float(divs[6].span.text.replace(',', ''))]
    return result


if __name__=="__main__":
    df = get_historical_data('amzn', '2017-9-6','2018-7-6')
    df.to_csv('df.csv')


