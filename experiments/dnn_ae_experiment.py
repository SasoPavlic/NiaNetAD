import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch import optim

from experiments.anomaly_detection import AnomalyDetection
from experiments.metrics import RMSE
from models.base import BaseAutoencoder


class DNNAEExperiment(LightningModule):
    def __init__(self,
                 dnn_ae_model: BaseAutoencoder,
                 params: dict,
                 n_features: int) -> None:
        super(DNNAEExperiment, self).__init__()

        # https://github.com/Lightning-AI/lightning/issues/4390#issuecomment-717447779
        # self.save_hyperparameters(logger=False)

        self.model = dnn_ae_model
        self.params = params
        self.n_features = n_features
        self.curr_device = None
        self.hold_graph = False
        # https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-and-devices
        self.testing_RMSE_metric = RMSE()
        self.test_RMSE = None
        self.AUC = None

        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = self.model.optimizer
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_signal, labels = batch
        self.curr_device = real_signal.device
        results = self.forward(real_signal)
        train_loss = self.model.loss_function(*results,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True, on_step=False,
                      on_epoch=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_signal, labels = batch
        self.curr_device = real_signal.device

        results = self.forward(real_signal)
        val_loss = self.model.loss_function(*results,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True, on_step=False,
                      on_epoch=True)
        # TODO add more metrics
        # https://github.com/Lightning-AI/metrics/issues/340#issuecomment-872073730
        return val_loss['loss']

    def on_fit_end(self) -> None:
        self.test_model()
        self.test_RMSE = self.testing_RMSE_metric.compute()

    def test_model(self):

        dataloader_iterator = iter(self.trainer.datamodule.test_dataloader())

        while True:
            try:
                data, target = next(dataloader_iterator)
            except StopIteration:
                break
            finally:
                self.testing_RMSE_metric.to(self.curr_device)
                results = self.model.generate(data, labels=target)
                self.testing_RMSE_metric.update(results[0], results[1])

    def on_train_end(self) -> None:
        self.get_AUC_metric()

    def get_AUC_metric(self):
        anomaly_detection = AnomalyDetection([0], [1])
        dataloader_iterator = iter(self.trainer.datamodule.test_dataloader())

        inputs = []
        outputs = []
        scores = []
        targets = []

        "Loop over dataset"
        for data, target in dataloader_iterator:
            data = data.to(self.curr_device)
            reconstructed, input = self.model.forward(data)

            "Loop over batch"
            for x, y, z in zip(input, reconstructed, target):
                inputs.append(x)
                outputs.append(y)
                targets.append(z.item())
                score = torch.sqrt(torch.sum((y - x) ** 2, dim=tuple(range(1, y.dim()))))
                scores.append(score.cpu().data.numpy().tolist())

        anomaly_detection.calculate_roc_auc_curve(targets, scores)

        self.AUC = anomaly_detection.AUC
