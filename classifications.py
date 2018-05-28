from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, SpectralClustering, Birch
from sklearn.metrics import silhouette_samples, silhouette_score
from preliminary_database import df_sales_creation
from exploratory_analysis import graph_component_silhouette
from db_analysis import zero_analysis, const_col_removal, pca_analysis, df_standardization, gap_optimalk
import numpy as np
import pandas as pd
import itertools
import time
from minisom import MiniSom
import sys


def feature_selection_power_set(df, basic_features):
    combinations_count = 0
    features = list(df)
    remaining_features = [x for x in features if x not in basic_features]
    for L in range(1, 2+1):
        for subset in itertools.combinations(remaining_features, L):
            print(basic_features + list(subset))
            combinations_count += 1
    print(combinations_count)


def cluster_application(df, n_clusters):

    start = time.time()
    scaled_matrix = df_standardization(df)

    # kmeans = [KMeans(n_clusters, n_init=30, n_jobs=1, algorithm='auto'), 'kmeans']
    kmeans = [MiniBatchKMeans(n_clusters, batch_size=10000, n_init=50, random_state=4), 'minibatchkmeans']
    # kmeans = [AffinityPropagation(n_clusters), 'affinity']  # Didn't finish, even with only 10% of data
    # kmeans = [SpectralClustering(n_clusters, n_jobs=1), 'spectral']  # UserWarning: Graph is not fully connected, spectral embedding may not work as expected.
    # kmeans = [Birch(n_clusters=n_clusters, compute_labels=True, threshold=0.3), 'birch']  #As fast as minibatchkmeans, but clusters are too big/small

    kmeans[0].fit(scaled_matrix)
    cluster_clients = kmeans[0].predict(scaled_matrix)
    silhouette_avg = silhouette_score(scaled_matrix, cluster_clients)
    print('\nFor', n_clusters, 'Clusters, the Avg. Silhouette Score is: %.3f' % silhouette_avg)

    _, counts = np.unique(cluster_clients, return_counts=True)
    print('Client Counts per cluster:', counts)
    sample_silhouette_values = silhouette_samples(scaled_matrix, cluster_clients)

    correct_clusters = 0
    for i in range(n_clusters):
        print(np.mean(sample_silhouette_values[cluster_clients == i]))

        if np.mean(sample_silhouette_values[cluster_clients == i]) >= silhouette_avg:
            correct_clusters += 1

    if correct_clusters == n_clusters:
        print('All Passed!')
        graph_component_silhouette(scaled_matrix, n_clusters, [-0.1, 1.0], len(scaled_matrix), sample_silhouette_values, silhouette_avg, cluster_clients, kmeans[1])
        print(list(df))
    else:
        print('Nope')

    print('Running time: %.3f' % (time.time() - start), 'seconds')

    graph_component_silhouette(scaled_matrix, n_clusters, [-0.1, 1.0], len(scaled_matrix), sample_silhouette_values, silhouette_avg, cluster_clients, kmeans[1], kmeans[0].labels_, kmeans[0].cluster_centers_)


def main():
    start = time.time()

    clustering = 1
    max_clusters = 20

    df = df_sales_creation(all_data=0).sample(frac=0.1, axis=0)
    # df = df_sales_creation(all_data=0)

    const_col_removal(df)

    basic_features = ['#PaidTransactions', 'MinPaidTransactions', 'MeanPaidTransactions', 'MaxPaidTransactions', 'SumPaidTransactions', 'New_VHE_Total_Cost', 'Used_VHE_Total_Cost', '#PSE_bought']
    # feature_selection_power_set(df, basic_features)
    # print(zero_analysis(df))

    # sys.exit()

    df = df[basic_features]
    matrix = df_standardization(df)
    n_clusters = gap_optimalk(matrix)

    if clustering:
        print('KMeans Application to DF Sales...')
        # for n_clusters in range(2, max_clusters+1):
        cluster_application(matrix, n_clusters=n_clusters)

        # for damping in np.arange(0.5, 1, 0.1):
            # cluster_application(df.sample(frac=0.1, axis=0), damping)

    print('\nRunning Time (Total): %.2f' % (time.time() - start), 'seconds')


if __name__ == '__main__':
    main()
