from tsfresh import extract_relevant_features, extract_features
import utilFeatExtr as util
import pandas as pd
import sys

listOut_train,series_train = util.adaptTimeSeries("../UCRArchive_2018/{0}/{0}_TRAIN.tsv".format(sys.argv[1]))

listOut_test,series_test = util.adaptTimeSeries("../UCRArchive_2018/{0}/{0}_TEST.tsv".format(sys.argv[1]))

# Questa è la funzione che vi estrae tutte le feature
all_features_train = extract_features(listOut_train, column_id='id', column_sort='time')

all_features_test = extract_features(listOut_test, column_id="id", column_sort="time")

# Elimino colonne con valori NaN
all_features_train = all_features_train.dropna(axis=1)
all_features_test = all_features_test.dropna(axis=1)

featureIntersection = all_features_train.columns.intersection(all_features_test.columns)

all_features_train = all_features_train.loc[:,featureIntersection]
all_features_test = all_features_test.loc[:,featureIntersection]

print(all_features_train.shape)
print(all_features_test.shape)

all_features_train.to_pickle("../pickle/feature_complete/TRAIN/{0}_TRAIN_FeatureComplete.pkl".format(sys.argv[1]))
all_features_test.to_pickle("../pickle/feature_complete/TEST/{0}_TEST_FeatureComplete.pkl".format(sys.argv[1]))

'''
# Per le rilevanti

# Questa è la funzione che vi estrae quelle interessanti
features_relevant_train = extract_relevant_features(listOut_train,series_train, column_id='id', column_sort='time')
features_relevant_test = extract_relevant_features(listOut_test,series_test, column_id='id', column_sort='time')

# Elimino colonne con valori NaN
features_relevant_train = features_relevant_train.dropna(axis=1)
features_relevant_test = features_relevant_test.dropna(axis=1)

featureIntersection = features_relevant_train.columns.intersection(features_relevant_test.columns)

features_relevant_train = features_relevant_train.loc[:,featureIntersection]
features_relevant_test = features_relevant_test.loc[:,featureIntersection]

features_relevant_train.to_pickle("../pickle/feature_rilevanti/TRAIN/{0}_TRAIN_FeatureRilevanti.pkl".format(sys.argv[1]))
features_relevant_test.to_pickle("../pickle/feature_rilevanti/TEST/{0}_TEST_FeatureRilevanti.pkl".format(sys.argv[1]))
'''
print("Salvato nel file")