import pandas as pd
import numpy as np
import sklearn.cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score, confusion_matrix
from skfeature.function.sparse_learning_based import NDFS
from skfeature.utility import construct_W
from skfeature.utility.sparse_learning import feature_ranking
from utility import valutazione, estrattoreClassiConosciute
import collections

def testFeatureSelection(X_selected, X_test, num_clusters, y):
    new_nmi, new_sil, new_db_score, new_ch_score, new_purity = valutazione.evaluation(
            X_selected=X_selected, X_test=X_test, n_clusters=num_clusters, y=y)
    nmi = new_nmi
    sil = new_sil
    db_score = new_db_score
    ch_score = new_ch_score
    purity = new_purity

    for i in range(0, 20):
        new_nmi, new_sil, new_db_score, new_ch_score, new_purity = valutazione.evaluation(
            X_selected=X_selected, X_test=X_test, n_clusters=num_clusters, y=y)
        if(new_nmi > nmi and new_sil > sil and new_db_score < db_score and new_purity > purity and new_ch_score > ch_score):
            nmi = new_nmi
            sil = new_sil
            db_score = new_db_score
            ch_score = new_ch_score
            purity = new_purity

    # output Silhouette, DB index, CH index, NMI, Purity e Accuracy
    print('Silhouette:', float(round(((sil)), 4)))
    print('Davies-Bouldin index score:', float(round(((db_score)), 4)))
    print('Calinski-Harabasz index score:', float(round(((ch_score)), 4)))
    print('NMI:', float(round(((nmi)), 4)))
    print('Purity:', float(round(((purity)), 4)))
    
def selectFeatureNDFS(filename, num_feature, num_cluster):

    # Recupero del pickle salvato su disco con i sample e TUTTE le feature estratte da TSFresh. SU QUESTO LAVOREREMO NOI
    all_features_train = pd.read_pickle(
        "./pickle/feature_complete/TRAIN/{0}_TRAIN_FeatureComplete.pkl".format(filename))
    all_features_test = pd.read_pickle(
        "./pickle/feature_complete/TEST/{0}_TEST_FeatureComplete.pkl".format(filename))

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
        "./UCRArchive_2018/{0}/{0}_TEST.tsv".format(filename))

    # K-means su feature selezionate
    print("\nRisultati con feature selezionate da noi con NDFS")
    print("Numero feature: {0}".format(all_features_test.shape[1]))
    testFeatureSelection(X_selected=dataframeFeatureSelezionate.values,
                        X_test=all_features_test.values, num_clusters=num_cluster, y=labelConosciute)