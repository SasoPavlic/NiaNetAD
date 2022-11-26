import numpy as np
from sklearn.metrics import accuracy_score

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

            self.AUC = abs(round(np.trapz(self.TPR_array, self.FPR_array), 3))


        except Exception as e:
            print(e)
            self.AUC = 0.0
