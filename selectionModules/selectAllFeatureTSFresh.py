import pandas as pd
import numpy as np
import sys
import sklearn.cluster
from skfeature.utility.sparse_learning import feature_ranking
from utility import valutazione, estrattoreClassiConosciute


def selectAllFeatureTSFresh(filename, num_cluster):

   # Recupero del pickle salvato su disco con i sample e TUTTE le feature estratte da TSFresh.
    all_features_train = pd.read_pickle(
        "./pickle/feature_complete/TRAIN/{0}_TRAIN_FeatureComplete.pkl".format(filename))
    all_features_test = pd.read_pickle(
        "./pickle/feature_complete/TEST/{0}_TEST_FeatureComplete.pkl".format(filename))

    # Estraggo le classi conosciute
    labelConosciute = estrattoreClassiConosciute.estraiLabelConosciute(
        "./UCRArchive_2018/{0}/{0}_TEST.tsv".format(filename))

    # K-means su dataframe estratto da TSFresh
    print("Risultati con tutte le feature estratte da TSFresh")
    print("Numero feature: {0}".format(all_features_test.shape[1]))
    nmi, sil, db_score, ch_score, purity = valutazione.evaluation(
    X_selected=all_features_train.values, X_test=all_features_test.values, n_clusters=num_cluster, y=labelConosciute)
    print('Silhouette:', float(round(((sil)), 4)))
    print('Davies-Bouldin index score:', float(round(((db_score)), 4)))
    print('Calinski-Harabasz index score:', float(round(((ch_score)), 4)))
    print('NMI:', float(round(((nmi)), 4)))
    print('Purity:', float(round(((purity)), 4)))