import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics import auc

from experiments.metrics import ConfusionMatrix


class AnomalyDetection(object):

    def __init__(self, valid_label, anomaly_label):
        self.valid_label = valid_label
        self.anomaly_label = anomaly_label
        self.acc_list = []

        self.metrics = []
        self.FPR_array = []
        self.TPR_array = []
        self.AUC = None

    def calculate_roc_auc_curve(self, targets, scores):
        # https://stackoverflow.com/questions/58894137/roc-auc-score-for-autoencoder-and-isolationforest

        try:
            fpr = dict()
            tpr = dict()
            thresholds = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], thresholds[i] = roc_curve(targets, scores)
                roc_auc[i] = auc(fpr[i], tpr[i])

            self.AUC = round(roc_auc[0], 3)

            # plt.figure()
            # lw = 2
            # plt.plot(
            #     fpr[1],
            #     tpr[1],
            #     color="darkorange",
            #     lw=lw,
            #     label="ROC curve (area = %0.2f)" % roc_auc[1],
            # )
            #
            # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel("False Positive Rate")
            # plt.ylabel("True Positive Rate")
            # plt.title("Receiver operating characteristic")
            # plt.legend(loc="lower right")
            # plt.show()

        except Exception as e:
            print(e)
            self.AUC = 0.0
