import os
from configparser import ConfigParser
import pymongo

curPath = os.path.dirname(__file__)
configPath = os.path.join(curPath, "database.ini")


def config(filename=configPath): # (filename=configPath, db_name="", collection_name=""):
    parser = ConfigParser()
    parser.read(filename)

    conn = pymongo.MongoClient(
        host=parser["MONGO"]["HOST"],
        port=int(parser["MONGO"]["PORT"]),
        username=parser["MONGO"]["USERNAME"],
        password=parser["MONGO"]["PASSWORD"],
    )
#     str_database_name = db_name
#     db = conn.get_database(str_database_name)

#     str_collection_name = collection_name
#     collection = db.get_collection(str_collection_name)
    return conn