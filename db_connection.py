# db_connection.py
import pymysql

def create_connection():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='demotst'
    )
    return connection
