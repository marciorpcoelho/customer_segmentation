import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.cm as cm
from preliminary_database import df_sales_creation
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


def save_fig(name, save_dir='output/'):
    # Saves plot in at least two formats, png and pdf
    plt.savefig(save_dir + str(name) + '.pdf')
    plt.savefig(save_dir + str(name) + '.png')


def total_transactions(out_dir):
    print('1 - Transaction Exploratory Analysis')

    my_dpi = 96
    all_data = 1  # defines if the plot uses only 2017 or all available data
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
        save_fig('1_transaction_distribution_all')
    elif not all_data:
        save_fig('1_transaction_distribution_2017')
    plt.show()


# 2
def graph_component_silhouette(X, n_clusters, lim_x, mat_size, sample_silhouette_values, silhouette_avg, clusters, method, cluster_labels=0, cluster_centers=0):
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    matplotlib.rc('patch', edgecolor='dimgray', linewidth=1)

    if cluster_labels:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(16, 8)
    elif not cluster_labels:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)

    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.8)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color='red', fontweight='bold', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))

        ax1.axvline(x=silhouette_avg, color='white', ls='--', lw=1.0)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

    ax1.set_xlabel("Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster label")

    if cluster_labels:
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Labeling the clusters
        centers = cluster_centers
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,s=50, edgecolor='k')

        # ax2.set_title("The visualization of the clustered data.", fontsize=10)
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data ""with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')

    plt.tight_layout()
    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-100")
    save_fig('2_customer_segmentation_' + str(method) + '_' + str(n_clusters) + '_cluster')
    # plt.show()


# 3
def pca_analysis_representation(pca, matrix):

    fig, ax = plt.subplots(figsize=(14, 5))

    sns.set(font_scale=1)
    plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid', label='cumulative explained variance')
    sns.barplot(np.arange(1, matrix.shape[1] + 1), pca.explained_variance_ratio_, alpha=0.5, color='g', label='individual explained variance')
    plt.xlim(-0.5, matrix.shape[1])

    ax.set_xticklabels([s if int(s.get_text()) % 2 == 0 else '' for s in ax.get_xticklabels()])

    plt.ylabel('Explained variance', fontsize=14)
    plt.xlabel('Principal components', fontsize=14)
    plt.title('PCA Analysis')
    plt.legend(loc='best', fontsize=13)

    plt.grid()
    plt.yticks(np.arange(0.0, 1.1, 0.1), [x*1. / 10 for x in range(0, 11, 1)])

    wm = plt.get_current_fig_manager()
    wm.window.wm_geometry("-1500-100")
    plt.tight_layout()
    plt.show()













def main():
    db_dir = 'sql_db/'
    out_dir = 'output/'

    # 1
    total_transactions(out_dir)


if __name__ == '__main__':
    main()
