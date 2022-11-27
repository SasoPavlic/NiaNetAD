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

    def find(self, input, reconstructed, target):
        """Compute ROC-AUC values and visualize it in plot
        """
        self.AUC = None
        self.metrics = []
        self.FPR_array = []
        self.TPR_array = []

        try:

            errors = []

            # loop over all original images and their corresponding
            # reconstructions
            for (x, y, z) in zip(input, reconstructed, target):
                # compute the mean squared error between the ground-truth image
                # and the reconstructed image, then add it to our list of errors
                result = (x - y) ** 2
                # result = result.detach().numpy()
                mse = np.mean(result.cpu().data.numpy().argmax())
                errors.append(mse)

            for quantile in np.linspace(0, 1, 100):
                threshold = np.quantile(errors, quantile)
                outliers_idx = np.where(np.array(errors) >= threshold)[0]
                quantile_instance_labels = np.array(target)[outliers_idx.astype(int)]

                metric = ConfusionMatrix(quantile, threshold, outliers_idx, quantile_instance_labels)

                metric.calculate_confusion_matrix(target, self.valid_label, self.anomaly_label)
                self.metrics.append(metric)

                self.FPR_array.append(metric.FPR)
                self.TPR_array.append(metric.TPR)

                """Calculating reconstruction accuracy per quantiles"""
                predicted_values = [self.anomaly_label if i in outliers_idx else self.valid_label for i, value in
                                    enumerate(target)]

                self.acc_list.append(accuracy_score(predicted_values, target))

            self.AUC = auc(self.TPR_array, self.FPR_array)



        except Exception as e:
            print(e)
            self.AUC = 0.0

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

            self.newAUC = round(roc_auc[0], 3)

            plt.figure()
            lw = 2
            plt.plot(
                fpr[1],
                tpr[1],
                color="darkorange",
                lw=lw,
                label="ROC curve (area = %0.2f)" % roc_auc[1],
            )

            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic")
            plt.legend(loc="lower right")
            plt.show()

        except Exception as e:
            print(e)
            self.newAUC = 0.0
