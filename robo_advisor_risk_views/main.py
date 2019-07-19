
from log_email import log_email
from sqlalchemy import create_engine
from ai_model_database_save import *
import os

@log_email()
def main():

    file_name = 'check_point'

    from_path = os.getcwd()
    to_path = os.getcwd() + '/check_point'
    connection = create_engine('mysql+pymysql://' + os.environ['MYSQL_USER'] + ':' + os.environ['MYSQL_PASSWORD'] + '@' + os.environ['MYSQL_HOST'] + ':' + os.environ['MYSQL_PORT']+'/ra_fttw')
    print('start to load the ai models ++++++++++++++++++++++++')
    load_from_database(zipfile_name=file_name, connection=connection, to_path=to_path)
    print('already loaded the ai models ++++++++++++++++++++++++\n')
    if os.path.exists('check_point'):
        print('the check_point file is exist!!!!!!')
    else:
        print('the check_point file dont exist!!!!!!')
    import ai.AI_prediciton.risk_classification_predict

    print('start to save the ai models ++++++++++++++++++++++++')
    save_to_database(file_name=file_name, connection=connection,from_path=from_path)
    print('already saved the ai models ++++++++++++++++++++++++')

    from multifactor import multi_ols

    import market_view.IndicatorWeight 
	

main()
