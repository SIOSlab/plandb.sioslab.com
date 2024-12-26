import pandas as pd
from sqlalchemy import create_engine
import pymysql
import glob
from sqlalchemy import text
import os
import sys 
import time

import logging

from update_plandb_method import *


"""
Update pre-existing database.
In this use case, update 2022 database.
Currently uses MySQL database: (databaseAfterUpdate)
Additionally uses MySQL database: (databaseDiffUpdate) to update the database
"""

start_time_update = time.time()
logging.info("update database start")

updateDatabaseTest("andrewchiu", "Password123!", "database2022before", 'databaseDiffUpdate', "databaseAfterUpdate")


# update

#TODO end time
end_time_update = time.time()
elapsed_time_update = end_time_update - start_time_update
logging.info(f"end update database elapsed time: {elapsed_time_update}")


