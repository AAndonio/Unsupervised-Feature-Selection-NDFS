import numpy as np
from utility import linear_assignment as la
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans


def calcolaPurity(labelConosciute, labels):
    contingencymatrix = contingency_matrix(labelConosciute, labels)
    purity = (np.sum(np.amax(contingencymatrix,axis = 0))/np.sum(contingencymatrix))
    return purity

def evaluation(X_selected, X_test, n_clusters, y):
    """
    This function calculates ARI, ACC and NMI of clustering results

    Input
    -----
    X_selected: {numpy array}, shape (n_samples, n_selected_features}
            input data on the selected features
    n_clusters: {int}
            number of clusters
    y: {numpy array}, shape (n_samples,)
            true labels

    Output
    ------
    nmi: {float}
        Normalized Mutual Information
    acc: {float}
        Accuracy
    """
    k_means = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                     tol=0.0001, precompute_distances=True, verbose=0,
                     random_state=None, copy_x=True, n_jobs=1)

    k_means.fit(X_selected)
    y_predict = k_means.predict(X_test)

    # calculate NMI
    nmi = normalized_mutual_info_score(y, y_predict, average_method='arithmetic')

    sil = silhouette_score(X_test, y_predict, metric="euclidean")
    db_score = davies_bouldin_score(X_test, y_predict)
    ch_score = calinski_harabasz_score(X_test, y_predict)
    purity = calcolaPurity(y, y_predict)

    return nmi, sil, db_score, ch_score, purity