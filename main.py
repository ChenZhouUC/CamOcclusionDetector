import os
import cv2
import configparser
import src.ConfigLoader as cl
import src.OcclusionDetector as od
import src.EvaluationMetrics as em


def loadingConfig(configFileRoot="./data/", configFileName="config.conf"):

    # configure loading
    configParser = configparser.ConfigParser()
    configParser = cl.ConfigLoader(
        configFileRoot, configFileName, configParser)
    print("ProblemSummary", "Description", cl.TopicValueExtractor(
        configParser, "ProblemSummary", "Description"), sep=" ==> ")

    # define data and label
    labeledDict = {}
    labelExposed = int(cl.TopicValueExtractor(
        configParser, "LabelDefinition", "Exposed"))
    repoExposed = cl.TopicValueExtractor(
        configParser, "DataLoader", "ExposedRepo")
    labelOccluded = int(cl.TopicValueExtractor(
        configParser, "LabelDefinition", "Occluded"))
    repoOccluded = cl.TopicValueExtractor(
        configParser, "DataLoader", "OccludedRepo")
    labeledDict[labelExposed] = [os.path.join(configFileRoot, repoExposed, d) for d in os.listdir(
        os.path.join(configFileRoot, repoExposed))]
    labeledDict[labelOccluded] = [os.path.join(configFileRoot, repoOccluded, d) for d in os.listdir(
        os.path.join(configFileRoot, repoOccluded))]

    return labeledDict


def evaluationProcess(labeledDict, size=(150, 100)):
    labelSeries = []
    probSeries = []
    for label in labeledDict.keys():
        print("...processing label: ", label, " ...")
        for data in labeledDict[label]:
            labelSeries.append(label)
            frame = cv2.imread(data,0)
            probSeries.append(od.OcclusionDetector(frame, size))
    # print(labelSeries, probSeries)
    auc_metrics = em.ROCMetrics(labelSeries, probSeries)
    return auc_metrics


if __name__ == "__main__":

    labeledDict = loadingConfig()
    auc_metrics = evaluationProcess(labeledDict)
    print("—————————————————————————————————————————")
    print("|    AUC Metrics Achieved: ", round(auc_metrics, 5), "    |")
    print("—————————————————————————————————————————")
