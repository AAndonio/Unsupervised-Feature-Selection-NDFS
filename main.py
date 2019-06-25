import pandas as pd
import sys
import numpy as np
import sklearn.cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score, confusion_matrix
import time
import estrattoreClassiConosciute
from skfeature.function.sparse_learning_based import NDFS
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking


# Opzioni per stampa di pandas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def stampaRisultati(dataframeIniziale, predict, labelConosciute, tempo):
    print("\nValutazione sul clustering senza tener conto delle label conosciute:\n")
    print("\tSilhouette score: {0} \n".format(silhouette_score(
        dataframeIniziale, predict, metric="euclidean")))
    print("\tDavies-Bouldin score: {0} \n".format(
        davies_bouldin_score(dataframeIniziale, predict)))
    print("\tCalinski-Harabasz score: {0} \n".format(
        calinski_harabasz_score(dataframeIniziale, predict)))

    print("Valutazione sul clustering tenenedo conto delle label conosciute:\n")
    print("\tNormalized mutual info score: {0} \n".format(
        normalized_mutual_info_score(labelConosciute, predict, average_method="arithmetic")))
    print("\tPurity: {0} \n".format(calcolaPurity(labelConosciute, predict)))
    print("Tempo: {0} \n".format(tempo))


def calcolaPurity(labelConosciute, labels):
    confusionMatrix = confusion_matrix(labelConosciute, labels)

    totale = 0

    for i in range(0, confusionMatrix.shape[0]):
        totale = totale + max(confusionMatrix[i])

    return totale/len(labels)


# Recupero del pickle salvato su disco con i sample e le feature rilevanti estratte da TSFresh. DA USARE PER CONFRONTO
relevant_features_train = pd.read_pickle(
    "./pickle/feature_rilevanti/TRAIN/{0}_TRAIN_FeatureRilevanti.pkl".format(sys.argv[1]))
relevant_features_test = pd.read_pickle(
    "./pickle/feature_rilevanti/TEST/{0}_TEST_FeatureRilevanti.pkl".format(sys.argv[1]))

# Recupero del pickle salvato su disco con i sample e TUTTE le feature estratte da TSFresh. SU QUESTO LAVOREREMO NOI
all_features_train = pd.read_pickle(
    "./pickle/feature_complete/TRAIN/{0}_TRAIN_FeatureComplete.pkl".format(sys.argv[1]))
all_features_test = pd.read_pickle(
    "./pickle/feature_complete/TEST/{0}_TEST_FeatureComplete.pkl".format(sys.argv[1]))

# Elimino colonne con valori NaN
all_features_train = all_features_train.dropna(axis=1)
all_features_test = all_features_test.dropna(axis=1)

# Costruisco matrice W da dare a NDFS
kwargs = {"metric": "euclidean", "neighborMode": "knn",
          "weightMode": "heatKernel", "k": 5, 't': 1}
W = construct_W.construct_W(all_features_train.values, **kwargs)

# Esecuzione dell'algoritmo NDFS. Otteniamo il peso delle feature per cluster.
featurePesate = NDFS.ndfs(all_features_train.values, n_clusters=20, W=W)

# ordinamento delle feature in ordine discendente
idx = feature_ranking(featurePesate)

num_feature = 10        # numero feature selezionate
num_cluster = 5         # number of clusters, it is usually set as the number of classes in the ground truth

idxSelected = idx[0:num_feature]   # seleziono il numero di feature che voglio

# Estraggo i nomi delle feature che ho scelto
nomiFeatureSelezionate = []

for i in idxSelected:
    nomiFeatureSelezionate.append(all_features_train.columns[i])

# Creo il dataframe con solo le feature che ho selezionato
dataframeFeatureSelezionate = all_features_train.loc[:, nomiFeatureSelezionate]

# Aggiusto anche il dataset di test con solo le feature scelte
all_features_test = all_features_test.loc[:, nomiFeatureSelezionate]

# Estraggo le classi conosciute
labelConosciute = estrattoreClassiConosciute.estraiLabelConosciute(
    "./UCRArchive_2018/{0}/{0}_TEST.tsv".format(sys.argv[1]))


# K-means su dataframe estratto da TSFresh
start = time.time()
kmeansTutte = sklearn.cluster.KMeans(n_clusters=num_cluster, init='k-means++', n_init=10, max_iter=300,
                                     tol=0.0001, precompute_distances=True, verbose=0,
                                     random_state=None, copy_x=True, n_jobs=1)

kmeansTutte.fit(relevant_features_train.values)
end = time.time()
tempo = end - start
labelsTutte = kmeansTutte.predict(relevant_features_test.values)

print("Valori k-means con tutte le feature estratte da TSFresh")
stampaRisultati(relevant_features_test.values,
                labelsTutte, labelConosciute, tempo)

# K-means su feature selezionate
start = time.time()
kmeansSelezionate = sklearn.cluster.KMeans(n_clusters=num_cluster, init='k-means++', n_init=10, max_iter=300,
                                           tol=0.0001, precompute_distances=True, verbose=0,
                                           random_state=None, copy_x=True, n_jobs=1)

kmeansSelezionate.fit(dataframeFeatureSelezionate.values)
end = time.time()
tempo = end - start
labels = kmeansSelezionate.predict(all_features_test.values)

print("===================================")
print("Valori k-means con solo feature selezionate")
stampaRisultati(all_features_test.values, labels, labelConosciute, tempo)
