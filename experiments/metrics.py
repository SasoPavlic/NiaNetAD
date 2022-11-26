from typing import Any

import torch
import torchmetrics
from torch import tensor, Tensor


class ConfusionMatrix(object):
    def __init__(self, quantile, threshold, outliers_idx, quantile_instance_labels):
        self.quantile = quantile
        self.threshold = threshold
        self.outliers_idx = outliers_idx
        self.quantile_instance_labels = quantile_instance_labels
        self.anomaly_count = None
        self.valid_count = None
        self.TP = None
        self.FN = None
        self.FP = None
        self.TN = None
        self.TPR = None
        self.FNR = None
        self.TNR = None
        self.FPR = None

    def evaluate_anomalies(self, TP, FN, FP, TN):
        """Compute recall, precision and F1-score
        Returns:
            Accuracy, Recall, Precision, F1-score
        """
        self.accuracy = ((TP + TN) / (TP + TN + FP + FN))
        self.recall = (TP / (TP + FN))
        self.precision = ((TP / (TP + FP)))
        self.F1 = 2 * ((self.precision * self.recall) / (self.precision + self.recall))

    def calculate_confusion_matrix(self, y_test, valid_label, anomaly_label):
        """Compute confusion matrix based on found anomalies in dataset
        """
        self.anomaly_count = sum(x in anomaly_label for x in y_test)
        self.valid_count = sum(x in valid_label for x in y_test)
        self.TP = sum(sum(x == anomaly_label for x in self.quantile_instance_labels))
        self.FN = self.anomaly_count - self.TP
        self.FP = len(self.outliers_idx) - self.TP
        self.TN = self.valid_count - self.FP

        self.TPR = (self.TP / (self.TP + self.FN))
        self.FNR = (self.FN / (self.TP + self.FN))
        self.TNR = (self.TN / (self.TN + self.FP))
        self.FPR = 1 - self.TNR

        self.evaluate_anomalies(self.TP, self.FN, self.FP, self.TN)


class RMSE(torchmetrics.Metric):
    # https: // www.pytorchlightning.ai / blog / torchmetrics - pytorch - metrics - built - to - scale
    def __init__(self, **kwargs: Any, ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_observations", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.n_observations += preds.numel()

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return torch.sqrt(self.sum_squared_error / self.n_observations)
