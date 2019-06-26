import pandas as pd
import sys
import numpy as np
import sklearn.cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score, confusion_matrix
import time
import estrattoreClassiConosciute
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking
import valutazione



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

num_feature = int(sys.argv[2])  # numero feature selezionate 
num_cluster = int(sys.argv[3])   # number of clusters, it is usually set as the number of classes in the ground truth


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
kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
W = construct_W.construct_W(all_features_train.values, **kwargs_W)

for i in range(0,5):

    # Esecuzione dell'algoritmo NDFS. Otteniamo il peso delle feature per cluster.
    featurePesate = lap_score.lap_score(all_features_train.values, W=W)

    # ordinamento delle feature in ordine discendente
    idx = lap_score.feature_ranking(featurePesate)

    print(idx)

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
nmi_total = 0
acc_total = 0
sil_total = 0
db_score_total = 0
ch_score_total = 0
purity_total = 0

for i in range(0, 20):
    nmi, acc, sil, db_score, ch_score, purity= valutazione.evaluation(X_selected=relevant_features_train.values,X_test = relevant_features_test.values, n_clusters=num_cluster, y=labelConosciute)
    nmi_total += nmi
    acc_total += acc
    sil_total += sil
    db_score_total += db_score
    ch_score_total += ch_score
    purity_total += purity

# output the average NMI and average ACC

print ('SIL:', float(sil_total)/20)
print ('DB SCORE:', float(db_score_total)/20)
print ('CH SCORE:', float(ch_score_total)/20)
print ('NMI:', float(nmi_total)/20)
print ('PURITY:', float(purity_total)/20)
print ('ACC:', float(acc_total)/20)

print("==================================")
# K-means su feature selezionate

nmi_total = 0
acc_total = 0
sil_total = 0
db_score_total = 0
ch_score_total = 0
purity_total = 0

for i in range(0, 20):
    nmi, acc, sil, db_score, ch_score, purity= valutazione.evaluation(X_selected=dataframeFeatureSelezionate.values,X_test = all_features_test.values, n_clusters=num_cluster, y=labelConosciute)
    nmi_total += nmi
    acc_total += acc
    sil_total += sil
    db_score_total += db_score
    ch_score_total += ch_score
    purity_total += purity

# output the average NMI and average ACC

print ('SIL:', float(sil_total)/20)
print ('DB SCORE:', float(db_score_total)/20)
print ('CH SCORE:', float(ch_score_total)/20)
print ('NMI:', float(nmi_total)/20)
print ('PURITY:', float(purity_total)/20)
print ('ACC:', float(acc_total)/20)

