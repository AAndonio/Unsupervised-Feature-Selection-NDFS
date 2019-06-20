import pandas as pd
import numpy as np
import sklearn.cluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, v_measure_score
from NDFS import ndfs

import estrattoreClassiConosciute

# Opzioni per stampa di pandas
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def stampaRisultati(dataframeIniziale, predict, labelConosciute):
    print("Valutazione sul clustering senza tener conto delle label conosciute:\n")
    print("\tSilhouette score: {0} \n".format(silhouette_score(dataframeIniziale, predict, metric="euclidean")))
    print("\tDavies-Bouldin score: {0} \n".format(davies_bouldin_score(dataframeIniziale, predict)))
    print("\tCalinski-Harabasz score: {0} \n".format(calinski_harabasz_score(dataframeIniziale, predict)))
    print(v_measure_score(labelConosciute, predict))


# Recupero del pickle salvato su disco con i sample e le feature rilevanti estratte da TSFresh
features_filtered_direct = pd.read_pickle("./pickle/ECG200.pkl")

# Esecuzione dell'algoritmo NDFS. Otteniamo il peso delle feature per cluster.
featurePesate = ndfs(features_filtered_direct.values, n_clusters=2)

# Creazione dataframe con feature e relativi pesi
dataframeFeaturePesate = pd.DataFrame(data = featurePesate[0:,0:], index = features_filtered_direct.columns)

# Calcolo della deviazione standard dei pesi per effettuare selezione. Si selezionano quelle con ds maggiore.
deviazioneStandardRighe = []
  
for i, j in dataframeFeaturePesate.iterrows(): 
    deviazioneStandardRighe.append(np.std(j, ddof = 1))

# Aggiungo colonna nel dataframe con la ds per la rispettiva riga. Ordino per ds in maniera discendente
dataframeFeaturePesate["sd"] = deviazioneStandardRighe
dataframeFeaturePesate = dataframeFeaturePesate.sort_values(["sd"], axis = 0, ascending = False)

# Calcolo del numero di feature dal selezionare. Qui usiamo il 70% delle feature estratte da TSFresh. Da far vedere.
numeroElementiSelezionati = int(round(len(dataframeFeaturePesate.index)*0.1))

# Estrazione nome feature rilevanti
nomiFeatureSelezionate = dataframeFeaturePesate.index[0:numeroElementiSelezionati]

# Selezione colonne delle feature rilevanti nel dataframe originale
dataframeFeatureSelezionate = features_filtered_direct.loc[:,nomiFeatureSelezionate]

labelConosciute = estrattoreClassiConosciute.estraiLabelConosciute("./UCRArchive_2018/ECG200/ECG200_TEST.tsv")

# labelConosciute = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

print(labelConosciute)

# K-means su dataframe estratto da TSFresh
kmeansTutte = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,
                                    tol=0.0001, precompute_distances=True, verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)

kmeansTutte.fit(features_filtered_direct)
labelsTutte = kmeansTutte.labels_                


print("Valori k-means con tutte le feature")
print(labelsTutte)
print()
stampaRisultati(features_filtered_direct, labelsTutte, labelConosciute)


# K-means su feature selezionate
kmeansSelezionate = sklearn.cluster.KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300,
                                    tol=0.0001, precompute_distances=True, verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)

kmeansSelezionate.fit(dataframeFeatureSelezionate)
labels = kmeansSelezionate.labels_ 

print("===================================")
print("Valori k-means con solo feature selezionate")
print(labels)
print()
stampaRisultati(dataframeFeatureSelezionate, labels, labelConosciute)






