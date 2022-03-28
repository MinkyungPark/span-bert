import os
import sys

import psycopg2
import pandas as pd
from configparser import ConfigParser

curPath = os.path.dirname(__file__)
configPath = os.path.join(curPath, "database.ini")


def config(filename=configPath, section="postgresql"):
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            "Section {0} not found in the {1} file".format(section, filename)
        )

    return db

def get_select(query):
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cursor = conn.cursor()

        cursor = conn.cursor()
        cursor.execute(query)
        record = cursor.fetchall()

        cursor.close()

        return record
    except (Exception, psycopg2.DatabaseError) as error:
        #         loger.info(error)
        print(error)
    finally:
        if conn is not None:
            cursor.close()
            conn.close()

            
def get_insert(query):
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
    except psycopg2.Error as e:
        print("Unable to connect")
        print(e.pgerror)
        print(e.diag.message_detail)
        sys.exit(1)
    else:
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        cursor.close()
        
def get_select_df(query):
    params = config()
    try:
        conn = psycopg2.connect(**params)
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    cursor = conn.cursor()

    try:
        # result = execute(query) # normal
        result = pd.read_sql_query(query, conn)  # pandas 이용
        return result
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            cursor.close()
            conn.close()