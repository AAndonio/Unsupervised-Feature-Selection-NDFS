from selectionModules import selectFeatureLapScore, selectFeatureNDFS, selectRelevantFeatureTSFresh
import sys

def main():

    filename = sys.argv[1]
    num_feature = int(sys.argv[2])
    num_cluster = int(sys.argv[3])

    selectRelevantFeatureTSFresh.selectRelevantFeatureTSFresh(filename, num_cluster)
    selectFeatureNDFS.selectFeatureNDFS(filename, num_feature, num_cluster)
    selectFeatureLapScore.selectFeatureLapScore(filename, num_feature, num_cluster)

main()