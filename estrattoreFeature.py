from tsfresh import extract_relevant_features, extract_features
import utilFeatExtr as util
import pandas as pd
from NDFS import ndfs

listOut,series = util.adaptTimeSeries("./UCRArchive_2018/ECG200/ECG200_TEST.tsv")
    
# Questa è la funzione che vi estrae quelle interessanti
features_filtered_direct = extract_relevant_features(listOut,series, column_id='id', column_sort='time')

print(features_filtered_direct)

features_filtered_direct.to_pickle("./ECG200.pkl")

print("Salvato nel file")