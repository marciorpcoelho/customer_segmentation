import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from preliminary_database import df_sales_creation
import sys
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


__author__ = 'mrpc'


def histogram(series, relative=1, bins=100):
    # Creates an histogram for the values at hand

    # max_value = (abs(series.min()) + abs(series.max()))
    quartile = series.quantile(0.75)
    n, bin_edge = np.histogram(series, bins=np.arange(0, quartile, round(quartile / bins)))
    if relative:
        n = n * 100 / n.sum()
    bincenters = 0.5 * (bin_edge[1:] + bin_edge[:-1])
    return bincenters, n


def total_transactions(db_dir, out_dir):
    print('1 - Transaction Exploratory Analysis')

    my_dpi = 96
    all_data = 0  # defines if the plot uses only 2017 or all available data
    df = df_sales_creation(all_data)
    transactiontypes = ['TA', 'RE', 'CM', 'FI', 'VN', 'VO']
    transactionlabel = ['Oficinas', 'Peças', 'CRM', 'Financeira', 'Viatura Nova', 'Viatura Usada']
    f, ax = plt.subplots(2, 2, figsize=(1500 / my_dpi, 800 / my_dpi), dpi=my_dpi)

    for type in transactiontypes:
        type_dist_paid = df[df[type + '_PaidTransactions'] > 0][type + '_PaidTransactions']
        type_dist_open = df[df[type + '_OpenTransactions'] > 0][type + '_OpenTransactions']

        if (type_dist_paid.iloc[type_dist_paid.nonzero()[0]]).shape[0]:
            bincenters_paid, n_paid = histogram(type_dist_paid, relative=0, bins=50)
            if type != 'VN' and type != 'VO':
                ax[0, 0].plot(bincenters_paid, n_paid, label=transactionlabel[transactiontypes.index(type)])
            if type == 'VN' or type == 'VO':
                ax[1, 0].plot(bincenters_paid, n_paid, label=transactionlabel[transactiontypes.index(type)])

        if (type_dist_open.iloc[type_dist_open.nonzero()[0]]).shape[0]:
            bincenters_open, n_open = histogram(type_dist_open, relative=0, bins=25)
            if type != 'VN' and type != 'VO':
                ax[0, 1].plot(bincenters_open, n_open, label=transactionlabel[transactiontypes.index(type)])
            if type == 'VN' or type == 'VO':
                ax[1, 1].plot(bincenters_open, n_open, label=transactionlabel[transactiontypes.index(type)])

    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-100")

    ax[0, 0].set_title('Paid Transaction Value Distribution Per Transaction Type')
    ax[0, 1].set_title('Open Transaction Value Distribution Per Transaction Type')
    for row, col in ax:
        row.legend(), col.legend()
        row.grid(), col.grid()
        row.set_xlabel('Transaction Value (€)'), row.set_ylabel('#Transactions')
        col.set_xlabel('Transaction Value (€)'), col.set_ylabel('#Transactions')

    plt.tight_layout()
    if all_data:
        plt.savefig(out_dir + '1_transaction_distribution_all.pdf')
    elif not all_data:
        plt.savefig(out_dir + '1_transaction_distribution_2017.pdf')
    plt.show()



















def main():
    db_dir = 'sql_db/'
    out_dir = 'output/'

    # 1
    total_transactions(db_dir, out_dir)


if __name__ == '__main__':
    main()
