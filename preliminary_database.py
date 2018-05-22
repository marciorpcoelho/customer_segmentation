import pandas as pd
from sql_conn import database_sql_retrieval
import db_analysis
import sys
pd.set_option('display.expand_frame_repr', False)

# df_ca_sales = database_sql_retrieval(database='BI_CA', view='View_VHE_Sales')
# df_ca_slr_open = database_sql_retrieval(database='BI_CA', view='View_SLR_OpenTransactions')
# df_ca_slr_paid = database_sql_retrieval(database='BI_CA', view='View_SLR_PaidTransactions')
# df_crp_sales = database_sql_retrieval(database='BI_CRP', view='View_VHE_Sales')
# df_crp_slr_open = database_sql_retrieval(database='BI_CRP', view='View_SLR_OpenTransactions')
# df_crp_slr_paid database_sql_retrieval(database='BI_CRP', view='View_SLR_PaidTransactions')


def df_sales_creation(all_data=0):
    # List of Tables to retrieve:
    dbs, views = ['BI_CA', 'BI_CRP'], ['View_VHE_Sales', 'View_SLR_OpenTransactions', 'View_SLR_PaidTransactions']
    for db in dbs:
        for view in views:
            database_sql_retrieval(database=db, view=view)

    database_sql_retrieval(database='BI_CA', view='View_PSE_Sales')
    print()

    df_open = pd.read_csv('sql_db/' + 'BI_CA_View_SLR_OpenTransactions.csv', index_col=0, dtype={'transactionFacility': object, 'transactionFacility2': object, 'SLR_Document': object, 'SLR_Account': object})
    df_paid = pd.read_csv('sql_db/' + 'BI_CA_View_SLR_PaidTransactions.csv', index_col=0, dtype={'transactionFacility': object, 'transactionFacility2': object, 'SLR_Document': object, 'SLR_Account': object})
    df_vhe = pd.read_csv('sql_db/' + 'BI_CA_View_VHE_Sales.csv', index_col=0, dtype={'transactionFacility': object, 'transactionFacility2': object, 'SLR_Document': object, 'SLR_Account': object})
    df_pse = pd.read_csv('sql_db/' + 'BI_CA_View_PSE_Sales.csv', index_col=0, dtype={'transactionFacility': object, 'transactionFacility2': object, 'SLR_Document': object, 'SLR_Account': object})

    dfs = [df_vhe, df_pse, df_paid, df_open]

    # Each client can have multiple SLR_Documents (receipts), but only one SLR_Account
    for df in dfs:
        df['dtTransaction'] = pd.to_datetime(df['dtTransaction'], format='%Y%m%d')
        df['day'] = df['dtTransaction'].dt.day
        df['month'] = df['dtTransaction'].dt.month
        df['year'] = df['dtTransaction'].dt.year

        if not all_data:
            df.drop(df[df['year'] != 2017].index, axis=0, inplace=True)  # Selecting the only year present for all the DB's, except OpenTransactions, in order to have data.
        df.drop(['NLR_Code'], axis=1, inplace=True)  # NLR_Code is the same for all DB as i'm using only CA data.

        try:
            df.drop(['Registration_Number'], axis=1, inplace=True)
        except KeyError:
            pass
        try:
            # removed = df[df['fiscalNumber'] == str(999999990)].shape
            df.drop(df[df['fiscalNumber'] == str(999999990)].index, axis=0, inplace=True)  # Removing the rows with a Fiscal Number of 999999990, probably from migration errors or non-available values
        except KeyError:
            pass

        df.rename(columns={'transactionFacility': 'centre', 'transactionFacility2': 'section'}, inplace=True)  # Renaming of these columns for easier understanding
        df.drop(df[df['SLR_Account'] == 0].index, axis=0, inplace=True)  # Removing these rows as they don't have a Fiscal Number

    df_open.dropna(subset=['fiscalNumber', 'section'], axis=0, inplace=True)  # As these values can't be imputed, they are removed
    df_paid.dropna(subset=['fiscalNumber', 'section'], axis=0, inplace=True)  # As these values can't be imputed, they are removed
    df_pse.dropna(subset=['SLR_Document', 'SLR_Account'], axis=0, inplace=True)  # As these values can't be imputed, they are removed

    df_paid = df_paid[df_paid['transactionValue'] >= 0]  # Removal of negative values of transactionValues, as they all have a positive value associated.
    df_open = df_open[df_open['transactionValue'] >= 0]  # Removal of negative values of transactionValues, as they all have a positive value associated.

    df_paid_grouped = df_paid.groupby(['SLR_Account'])
    df_open_grouped = df_open.groupby(['SLR_Account'])

    df_sales = pd.DataFrame({'#VHE_bought': df_vhe.groupby(['SLR_Account']).size(),  # Counts and aggregates the number of new car sales
                             '#PSE_bought': df_pse.groupby(['SLR_Account']).size(),  # Counts and aggregates the number of used car sales
                             '#PaidTransactions': df_paid_grouped.size(),  # Counts and aggregates the number of paid transactions
                             'TA_PaidTransactions': df_paid[df_paid['transactionType'] == 'TA'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'RE_PaidTransactions': df_paid[df_paid['transactionType'] == 'RE'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'CM_PaidTransactions': df_paid[df_paid['transactionType'] == 'CM'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'FI_PaidTransactions': df_paid[df_paid['transactionType'] == 'FI'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'VN_PaidTransactions': df_paid[df_paid['transactionType'] == 'VN'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'VO_PaidTransactions': df_paid[df_paid['transactionType'] == 'VO'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'PaidTotalTransactions': df_paid_grouped['transactionValue'].sum(),  #Sums and aggregates the total value of all paid transactions
                             'MeanPaidTransactions': df_paid_grouped['transactionValue'].mean(),  #Calculates the average value of each transaction
                             '#OpenTransactions': df_open_grouped.size(),  # Counts and aggregates the number of open transactions
                             'TA_OpenTransactions':df_open[df_open['transactionType'] == 'TA'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'RE_OpenTransactions':df_open[df_open['transactionType'] == 'RE'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'CM_OpenTransactions':df_open[df_open['transactionType'] == 'CM'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'FI_OpenTransactions':df_open[df_open['transactionType'] == 'FI'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'VN_OpenTransactions':df_open[df_open['transactionType'] == 'VN'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'VO_OpenTransactions':df_open[df_open['transactionType'] == 'VO'].groupby(['SLR_Account'])['transactionValue'].sum().astype(float),
                             'OpenTotalTransactions': df_open_grouped['transactionValue'].sum(),  # Sums and aggregates the total value of all open transactions
                             'MeanOpenTransactions': df_open_grouped['transactionValue'].mean(),
                             '#Transactions': df_paid_grouped.size().add(df_open_grouped.size(), fill_value=0)}  # Counts and aggregates the total number of transactions
                            ).fillna(0)

    df_sales.index.rename('SLR_Account', inplace=True)


    # After looking at df_sales, 1773 rows have no value in PaidTotalTransaction. This does not make sense for now, unless it refers to previous transactions (before 2017). Will need to look at this. But for now, they are removed:
    df_sales.drop(df_sales[df_sales['PaidTotalTransactions'] == 0].index, axis=0, inplace=True)

    return df_sales


def main():
    df_sales_creation()


if __name__ == '__main__':
    main()