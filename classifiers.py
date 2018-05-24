import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
pd.set_option('display.expand_frame_repr', False)


class ClassFit(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def grid_search(self, parameters, Kfold):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=Kfold)

    def grid_fit(self, X, Y):
        self.grid.fit(X, Y)

    def grid_predict(self, X, Y):
        self.predictions = self.grid.predict(X)
        print("Precision: {:.2f} % ".format(100 * metrics.accuracy_score(Y, self.predictions)))


class KMeansCluster(object):
    def __init__(self, clf_cluster, params=None):
        if params:
            self.clf_cluster = clf_cluster(**params)
        else:
            self.clf_cluster = clf_cluster()

    def kmeans_fit(self, X):
        self.clf_cluster.fit(X)

    def kmeans_predict(self, X):
        self.clf_cluster.predict(X)

    def silhouette_score(self, X, clusters):
        self.shilhouette_avg = silhouette_score(X, clusters)
        print('For ', 'clusters, the average silhouette score is:', self.shilhouette_avg)