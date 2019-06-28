from selectionModules import selectFeatureLapScore, selectFeatureNDFS, selectRelevantFeatureTSFresh, selectAllFeatureTSFresh
import sys

def main():

    num_cluster_dizionario = {
        "ECG200": 2,
        "ECG5000": 5,
        "FordA": 2,
        "FordB": 2,
        "ChlorineConcentration": 3,
        "PhalangesOutlinesCorrect": 3, 
        "RefrigerationDevices": 3,
        "TwoLeadECG": 2,
        "TwoPatterns": 4
    }

    filename = sys.argv[1]
    num_feature = int(sys.argv[2])
    num_cluster = num_cluster_dizionario[filename]

    selectAllFeatureTSFresh.selectAllFeatureTSFresh(filename, num_cluster)
    selectRelevantFeatureTSFresh.selectRelevantFeatureTSFresh(filename, num_cluster)
    selectFeatureNDFS.selectFeatureNDFS(filename, num_feature, num_cluster)
    selectFeatureLapScore.selectFeatureLapScore(filename, num_feature, num_cluster)

main()