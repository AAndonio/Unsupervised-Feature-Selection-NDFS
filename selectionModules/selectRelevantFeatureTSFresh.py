import pandas as pd
import numpy as np
import sys
import sklearn.cluster
from skfeature.utility.sparse_learning import feature_ranking
from utility import valutazione, estrattoreClassiConosciute


def selectRelevantFeatureTSFresh(filename, num_cluster):
    # Recupero del pickle salvato su disco con i sample e le feature rilevanti estratte da TSFresh. DA USARE PER CONFRONTO
    relevant_features_train = pd.read_pickle(
        "./pickle/feature_rilevanti/TRAIN/{0}_TRAIN_FeatureRilevanti.pkl".format(filename))
    relevant_features_test = pd.read_pickle(
        "./pickle/feature_rilevanti/TEST/{0}_TEST_FeatureRilevanti.pkl".format(filename))

    # Estraggo le classi conosciute
    labelConosciute = estrattoreClassiConosciute.estraiLabelConosciute(
        "./UCRArchive_2018/{0}/{0}_TEST.tsv".format(filename))

    # K-means su dataframe estratto da TSFresh
    print("Risultati con feature rilevanti estratte da TSFresh")
    nmi, sil, db_score, ch_score, purity = valutazione.evaluation(
    X_selected=relevant_features_train.values, X_test=relevant_features_test.values, n_clusters=num_cluster, y=labelConosciute)
    print('Silhouette:', float(sil))
    print('Davies-Bouldin index score:', float(db_score))
    print('Calinski-Harabasz index score:', float(ch_score))
    print('NMI:', float(round(((nmi)), 4)))
    print('Purity:', float(purity))