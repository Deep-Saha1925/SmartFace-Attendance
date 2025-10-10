import mysql.connector
from config import MYSQL_PASSWORD, MYSQL_USERNAME, MYSQL_HOST, MYSQL_DATABASE

def get_connection():
    connection = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USERNAME,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    
    return connection