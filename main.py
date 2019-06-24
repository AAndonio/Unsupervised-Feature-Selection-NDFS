import pandas as pd
import sys
import numpy as np
import sklearn.cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score, confusion_matrix
from NDFS import ndfs
import time

import estrattoreClassiConosciute

# Opzioni per stampa di pandas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def stampaRisultati(dataframeIniziale, predict, labelConosciute, tempo):
    print("\nValutazione sul clustering senza tener conto delle label conosciute:\n")
    print("\tSilhouette score: {0} \n".format(silhouette_score(dataframeIniziale, predict, metric="euclidean")))
    print("\tDavies-Bouldin score: {0} \n".format(davies_bouldin_score(dataframeIniziale, predict)))
    print("\tCalinski-Harabasz score: {0} \n".format(calinski_harabasz_score(dataframeIniziale, predict)))

    print("Valutazione sul clustering tenenedo conto delle label conosciute:\n")
    print("\tNormalized mutual info score: {0} \n".format(normalized_mutual_info_score(labelConosciute, predict, average_method="arithmetic")))
    print("\tPurity: {0} \n".format(calcolaPurity(labelConosciute, predict)))
    print("Tempo: {0} \n".format(tempo))      

def calcolaPurity(labelConosciute, labels):
        confusionMatrix = confusion_matrix(labelConosciute, labels)

        totale = 0

        for i in range(0,confusionMatrix.shape[0]):
                totale = totale + max(confusionMatrix[i])

        return totale/len(labels)


# Recupero del pickle salvato su disco con i sample e le feature rilevanti estratte da TSFresh. DA USARE PER CONFRONTO
relevant_features_train = pd.read_pickle("./pickle/feature_rilevanti/TRAIN/{0}_TRAIN_FeatureRilevanti.pkl".format(sys.argv[1]))
relevant_features_test = pd.read_pickle("./pickle/feature_rilevanti/TEST/{0}_TEST_FeatureRilevanti.pkl".format(sys.argv[1]))

# Recupero del pickle salvato su disco con i sample e TUTTE le feature estratte da TSFresh. SU QUESTO LAVOREREMO NOI
all_features_train = pd.read_pickle("./pickle/feature_complete/TRAIN/{0}_TRAIN_FeatureComplete.pkl".format(sys.argv[1])) 
all_features_test = pd.read_pickle("./pickle/feature_complete/TEST/{0}_TEST_FeatureComplete.pkl".format(sys.argv[1])) 


all_features_train = all_features_train.dropna(axis=1)
all_features_test = all_features_test.dropna(axis=1)

# Esecuzione dell'algoritmo NDFS. Otteniamo il peso delle feature per cluster.
featurePesate = ndfs(all_features_train.values, n_clusters=2)

# Creazione dataframe con feature e relativi pesi
dataframeFeaturePesate = pd.DataFrame(data = featurePesate[0:,0:], index = all_features_train.columns)

# Calcolo della deviazione standard dei pesi per effettuare selezione. Si selezionano quelle con ds maggiore.
deviazioneStandardRighe = []
  
for i, j in dataframeFeaturePesate.iterrows(): 
    deviazioneStandardRighe.append(np.std(j, ddof = 1))

# Aggiungo colonna nel dataframe con la ds per la rispettiva riga. Ordino per ds in maniera discendente
dataframeFeaturePesate["sd"] = deviazioneStandardRighe
dataframeFeaturePesate = dataframeFeaturePesate.sort_values(["sd"], axis = 0, ascending = False)

# Calcolo del numero di feature dal selezionare. Qui usiamo il 70% delle feature estratte da TSFresh. Da far vedere.
numeroElementiSelezionati = int(round(len(dataframeFeaturePesate.index)*0.5))

# Indice dell'80% delle feature più importanti su NDFS secondo l'analisi di Pareto, effettuato sulle deviazioni standard

'''
somma = sum(dataframeFeaturePesate["sd"]) * 0.8
Pareto = 0
contatoreFeature = 0
for i in dataframeFeaturePesate["sd"]:
    Pareto = Pareto + i
    contatoreFeature = contatoreFeature + 1
    if (Pareto >= somma):
        print("Numero di feature senza Pareto: " + str(len(dataframeFeaturePesate["sd"])))
        print("Numero di feature senza Pareto (80%): " + str(round(len(dataframeFeaturePesate["sd"]) * 0.8)))
        print("Numero di feature secondo Pareto: " + str(contatoreFeature) + "\n")
        break

numeroElementiSelezionati = contatoreFeature

'''

# Estrazione nome feature rilevanti
nomiFeatureSelezionate = dataframeFeaturePesate.index[0:numeroElementiSelezionati]

# Selezione colonne delle feature rilevanti nel dataframe originale
dataframeFeatureSelezionate = all_features_train.loc[:,nomiFeatureSelezionate]
all_features_test = all_features_test.loc[:,nomiFeatureSelezionate]


labelConosciute = estrattoreClassiConosciute.estraiLabelConosciute("./UCRArchive_2018/{0}/{0}_TEST.tsv".format(sys.argv[1]))

start = time.time()

# K-means su dataframe estratto da TSFresh
kmeansTutte = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,
                                    tol=0.0001, precompute_distances=True, verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)

kmeansTutte.fit(relevant_features_train)
end = time.time()
tempo = end - start
labelsTutte = kmeansTutte.predict(relevant_features_test)              


print("Valori k-means con tutte le feature estratte da TSFresh")
stampaRisultati(relevant_features_test, labelsTutte, labelConosciute, tempo)

start = time.time()
# K-means su feature selezionate
kmeansSelezionate = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,
                                    tol=0.0001, precompute_distances=True, verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)

kmeansSelezionate.fit(dataframeFeatureSelezionate)
end = time.time()
tempo = end - start
labels = kmeansSelezionate.predict(all_features_test)

print("===================================")
print("Valori k-means con solo feature selezionate")
print(len(labels))


stampaRisultati(all_features_test, labels, labelConosciute, tempo)