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
import valutazione
import collections


# Opzioni per stampa di pandas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def testFeatureSelection(X_selected, X_test, num_clusters, y):
    new_nmi, new_acc, new_sil, new_db_score, new_ch_score, new_purity = valutazione.evaluation(
            X_selected=X_selected, X_test=X_test, n_clusters=num_cluster, y=y)
    nmi = new_nmi
    acc = new_acc
    sil = new_sil
    db_score = new_db_score
    ch_score = new_ch_score
    purity = new_purity

    for i in range(0, 20):
        new_nmi, new_acc, new_sil, new_db_score, new_ch_score, new_purity = valutazione.evaluation(
            X_selected=X_selected, X_test=X_test, n_clusters=num_cluster, y=y)
        if(new_nmi > nmi and new_acc > acc and new_sil > sil and new_db_score < db_score and new_purity > purity and new_ch_score > ch_score):
            nmi = new_nmi
            acc = new_acc
            sil = new_sil
            db_score = new_db_score
            ch_score = new_ch_score
            purity = new_purity

    # output Silhouette, DB index, CH index, NMI, Purity e Accuracy
    print('Silhouette:', float(sil))
    print('Davies-Bouldin index score:', float(db_score))
    print('Calinski-Harabasz index score:', float(ch_score))
    print('NMI:', float(round(((nmi)), 4)))
    print('Purity:', float(purity))
    print('Accuracy:', float(acc))


num_feature = int(sys.argv[2]) # numero feature selezionate
num_cluster = int(sys.argv[3]) # numero di cluster, va messo uguale al numero di classi conosciute (quelle nel tsv per intenderci)


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

# Eseguo più volte NDFS in modo da ottenere davvero le feature migliori
dizionarioOccorrenzeFeature = {}

for i in range(0, 10):

    # Esecuzione dell'algoritmo NDFS. Otteniamo il peso delle feature per cluster.
    featurePesate = NDFS.ndfs(all_features_train, n_clusters=20, W=W)

    # ordinamento delle feature in ordine discendente
    idx = feature_ranking(featurePesate)

    # prendo il numero di feature scelte
    idxSelected = idx[0:num_feature]

    # aggiorno il numero di occorrenze di quella feature nel dizionario
    for feature in idxSelected:
        if feature in dizionarioOccorrenzeFeature:
            dizionarioOccorrenzeFeature[feature] = dizionarioOccorrenzeFeature[feature]+1
        else:
            dizionarioOccorrenzeFeature[feature] = 1

# Ordino il dizionario in maniera discendente in modo da avere la feature che compare più volte all'inizio.
# Qui abbiamo un dizionario contenente tupla (nomeFeature, numeroOccorrenze)
dizionarioOccorrenzeFeature_sorted = sorted(dizionarioOccorrenzeFeature.items(), key=lambda kv: -kv[1])

# Metto tutti in nomi delle feature presenti in nel dizionario in un array
featureFrequenti = []
for key, value in dizionarioOccorrenzeFeature_sorted:
    featureFrequenti.append(key)

# seleziono il numero di feature che voglio
idxSelected = featureFrequenti[0:num_feature]

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
print("Risultati con feature rilevanti estratte da TSFresh")
nmi, acc, sil, db_score, ch_score, purity = valutazione.evaluation(
X_selected=relevant_features_train.values, X_test=relevant_features_test.values, n_clusters=num_cluster, y=labelConosciute)
print('Silhouette:', float(sil))
print('Davies-Bouldin index score:', float(db_score))
print('Calinski-Harabasz index score:', float(ch_score))
print('NMI:', float(round(((nmi)), 4)))
print('Purity:', float(purity))
print('Accuracy:', float(acc))

# K-means su feature selezionate
print("Risultati con feature selezionate da noi con NDFS")
testFeatureSelection(X_selected=dataframeFeatureSelezionate.values,
                     X_test=all_features_test.values, num_clusters=num_cluster, y=labelConosciute)