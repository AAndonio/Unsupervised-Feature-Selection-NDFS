from tsfresh import extract_relevant_features, extract_features
import utilFeatExtr as util
import pandas as pd
import sys

listOut_train,series_train = util.adaptTimeSeries("./UCRArchive_2018/{0}/{0}_TRAIN.tsv".format(sys.argv[1]))

listOut_test,series_test = util.adaptTimeSeries("./UCRArchive_2018/{0}/{0}_TEST.tsv".format(sys.argv[1]))

# Questa è la funzione che vi estrae quelle interessanti
# features_relevant_train = extract_relevant_features(listOut_train,series_train, column_id='id', column_sort='time')
features_relevant_test = extract_relevant_features(listOut_test,series_test, column_id='id', column_sort='time')

print(len(features_relevant_test.columns))


'''
featureIntersection = features_relevant_train.columns.intersection(features_relevant_test.columns)

features_relevant_train = features_relevant_train.loc[:,featureIntersection]
features_relevant_test = features_relevant_test.loc[:,featureIntersection]

features_relevant_train.to_pickle("./pickle/feature_rilevanti/TRAIN/{0}_TRAIN_FeatureRilevanti.pkl".format(sys.argv[1]))
features_relevant_test.to_pickle("./pickle/feature_rilevanti/TEST/{0}_TEST_FeatureRilevanti.pkl".format(sys.argv[1]))

print("Salvato nel file")
'''