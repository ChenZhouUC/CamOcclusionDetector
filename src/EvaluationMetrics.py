import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def ROCMetrics(labelSeries, probSeries, posLabel=1, saving=False):
    fpr, tpr, thresholds = roc_curve(
        labelSeries, probSeries, pos_label=posLabel)
    auc_metrics = auc(fpr, tpr)
    # depict the ROC metrics
    sns.set()
    _ = sns.lineplot(x=fpr, y=tpr)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve | AUC="+str(round(auc_metrics, 2)))
    if saving:
        plt.savefig('./plots/ROC_Metrics.png')
    plt.show()

    return auc_metrics


def HardCaseMining(caseSeries, scoreSeries, flag, topN=10):
    if flag == "pos":
        flag = 1
    else:
        flag = -1
    idx = np.argsort(flag*np.array(scoreSeries))[:topN]
    return np.array(caseSeries)[idx], np.array(scoreSeries)[idx]


if __name__ == "__main__":

    labelSeries = np.array([1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2])
    probSeries = np.array([0.1, 0.4, 0.4, 0.3, 0.5,
                           0.4, 0.6, 0.7, 0.8, 0.5])   # probability of prediction as positive

    ROCMetrics(labelSeries, probSeries)
