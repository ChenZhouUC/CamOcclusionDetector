import os
import cv2
import numpy as np
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


def evaluationProcess(labeledDict, size=(150, 100), super_pixel_set=(15, 10), posLabel=1,
                      saving_output=True, hard_case_topN=10, plotSaveName="ROC_Metrics.png"):
    labelSeries = []
    probSeries = []
    caseSeries = {}
    for label in labeledDict.keys():
        print("...processing label: ", label, " ...")
        tmp_label = []
        tmp_prob = []
        for data in labeledDict[label]:
            tmp_label.append(label)
            frame = cv2.imread(data)
            tmp_prob.append(od.OcclusionDetector(
                frame, size, super_pixel_set))
            # tmp_prob.append(od.SaturationDetector(frame,size))
        labelSeries.extend(tmp_label)
        probSeries.extend(tmp_prob)
        if label == posLabel:
            caseSeries["pos"] = {"case": labeledDict[label],
                                 "prob": tmp_prob}
        else:
            caseSeries["neg"] = {"case": labeledDict[label],
                                 "prob": tmp_prob}

    # print(labelSeries, probSeries)
    pos_hc = em.HardCaseMining(
        caseSeries["pos"]["case"], caseSeries["pos"]["prob"], "pos", topN=hard_case_topN)
    neg_hc = em.HardCaseMining(
        caseSeries["neg"]["case"], caseSeries["neg"]["prob"], "neg", topN=hard_case_topN)
    auc_metrics = em.ROCMetrics(
        labelSeries, probSeries, posLabel=posLabel, saving=saving_output, plotSaveName=plotSaveName)

    return auc_metrics, pos_hc, neg_hc


def showHardCases(hcResult, hcType, showSize=(600, 400)):
    hcList, hcScore = hcResult
    textColor = (0, 255, 0) if hcType == "NEGATIVE" else (0, 0, 255)
    for idx, hc in enumerate(hcList):
        hc_frame = cv2.imread(hc)
        hc_frame = cv2.resize(hc_frame, showSize)
        cv2.putText(hc_frame, str(round(hcScore[idx], 2)), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2)
        cv2.imshow(hcType, hc_frame)
        cv2.waitKey(0)
    print(str(len(hcList))+" hard cases for "+hcType+" showed...")
    cv2.destroyWindow(hcType)


if __name__ == "__main__":

    labeledDict = loadingConfig()
    auc_metrics, pos_hc, neg_hc = evaluationProcess(labeledDict, hard_case_topN=20,
                                                    super_pixel_set=(25, 20),
                                                    plotSaveName="ROC_Metrics.png")
    print("—————————————————————————————————————————")
    print("|    AUC Metrics Achieved: ", round(auc_metrics, 5), "    |")
    print("—————————————————————————————————————————")
    showHardCases(pos_hc, "POSITIVE")
    showHardCases(neg_hc, "NEGATIVE")
