__author__ = 'mrpc'
import pyodbc
import pandas as pd
import time
import os


def database_sql_retrieval(database, view):

    # Establish Connection to DW and retrieves the table from a specific view from a specific table;
    # After connection, saves the table as a .csv file;
    # try:
    if os.path.isfile('sql_db/' + database + '_' + view + '.csv'):
        # df = pd.read_csv('sql_db/' + database + '_' + view + '.csv', index_col=0)
        print('Dataframe for view ' + view + ' from database ' + database + ' already exists...')
    # except FileNotFoundError:
    else:
        start = time.time()

        print('Connecting to SQL DW...')
        conn = pyodbc.connect('DSN=RCG_DW;UID=RCG_DISC;PWD=rcg_disc;DATABASE=' + database)
        query = """SELECT * from RCG.""" + view
        cursor = conn.cursor()

        print('Creating dataframe for view ' + view + ' from database ' + str(database) + '...')
        cursor.execute(query)

        df = pd.DataFrame.from_records(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        df.to_csv('sql_db/' + database + '_' + view + '.csv')

        print('Elapsed time: %.2f' % (time.time() - start), 'seconds...')

    # return df
