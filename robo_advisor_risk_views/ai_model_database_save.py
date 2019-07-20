import mysql.connector
import os
import zipfile
from datetime import datetime
from mysql.connector import Error
from mysql.connector import errorcode

from sqlalchemy import create_engine

def make_zip_file(file_name,file_path):
    zip_file = zipfile.ZipFile(file_name + '.zip', 'w')
    file_name = file_path+'/'+file_name
    for i in os.listdir(file_name):
        zip_file.write(filename=os.path.join(file_name, i), arcname=i,compress_type=zipfile.ZIP_LZMA)
    zip_file.close()

def unzip_file(zipfile_name,to_file_path):
    f = zipfile.ZipFile(zipfile_name + '.zip', 'r')
    for file in f.namelist():
        f.extract(file, to_file_path)

    for i in os.listdir(to_file_path):
        print(i,'\n')

def convert_to_binary_data(file_name):

    with open(file_name, 'rb') as file:
        binaryData = file.read()
    return binaryData

def write_file(data, filename):
    """
    Convert binary data to proper format and write it on Hard Disk
    :param data:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as file:
        file.write(data)

def save_to_database(file_name,connection,from_path):

    today = datetime.today()

    cursor = connection.connect()

    sql_insert_blob_query = """ INSERT INTO `ai_model`(`date`,`model`) VALUES (%s,%s)"""

    make_zip_file(file_name=file_name,file_path=from_path)
    size = os.path.getsize('check_point.zip')
    print('the size of check_point file is ============ ',size)

    bi_zipfile = convert_to_binary_data(file_name=file_name+'.zip')

    insert_blob_tuple = (today, bi_zipfile)
    cursor.execute("""DELETE FROM `ai_model`""")

    cursor.execute(sql_insert_blob_query, insert_blob_tuple)

def load_from_database(zipfile_name,connection,to_path):
    print("Reading BLOB data from table")
    try:

        cursor = connection.connect()
        sql_fetch_blob_query = """SELECT * from `ai_model`"""

        record = cursor.execute(sql_fetch_blob_query).fetchone()

        if record != None:
            ai_model_zipfile = record[2]

            write_file(ai_model_zipfile, zipfile_name+'.zip')
            if os.path.exists(zipfile_name+'.zip'):
                print('the check_point zip file is exist!!!!')

            unzip_file(zipfile_name, to_file_path=to_path)
        else:
            pass
    except mysql.connector.Error as error:
        connection.rollback()
        print("Failed to read BLOB data from MySQL table {}".format(error))

if __name__ == '__main__':
    file_name = 'check_point'

    connection = create_engine('mysql+pymysql://' + os.environ['MYSQL_USER'] + ':' + os.environ['MYSQL_PASSWORD'] + '@' + os.environ['MYSQL_HOST'] + ':' + os.environ['MYSQL_PORT']+'/ra_fttw')
    from_path = os.getcwd()+'/new/new1'
    to_path = os.getcwd()+'/new/new1/check_point'

    load_from_database(zipfile_name=file_name,connection=connection,to_path=to_path)
