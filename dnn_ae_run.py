import argparse
import math
import uuid
from pathlib import Path

import torch
import yaml
from niapy.algorithms.basic import ParticleSwarmAlgorithm, DifferentialEvolution, FireflyAlgorithm, GeneticAlgorithm
from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch import Trainer
from tabulate import tabulate

from dataloaders.tabular import TabularDataset
from experiments.dnn_ae_experiment import DNNAEExperiment
from models.dnn_ae import Autoencoder
from niapy_extension.wrapper import *
from storage.database import SQLiteConnector

RUN_UUID = uuid.uuid4().hex
parser = argparse.ArgumentParser(description='Generic runner for DNN AE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/dnn_ae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

config['logging_params']['save_dir'] += RUN_UUID + '/'
Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

early_stop_callback = EarlyStopping(monitor=config['early_stop']['monitor'],
                                    min_delta=config['early_stop']['min_delta'],
                                    patience=config['early_stop']['patience'],
                                    verbose=False,
                                    check_finite=True,
                                    mode="max")

conn = SQLiteConnector(config['logging_params']['db_storage'], f"solutions")  # _{RUN_UUID}")
seed_everything(config['exp_params']['manual_seed'], True)

datamodule = TabularDataset(**config["data_params"], pin_memory=True)
datamodule.setup()


class DNNAEArchitecture(ExtendedProblem):

    def __init__(self, dimension):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.iteration = 0

    def _evaluate(self, solution, alg_name):
        print("=================================================================================================")
        print(f"ITERATION: {self.iteration}")
        print(f"SOLUTION : {solution}")
        self.iteration += 1

        model = Autoencoder(solution, **config)
        existing_entry = conn.get_entries(hash_id=model.hash_id)
        path = config['logging_params']['save_dir'] + str(self.iteration) + "_" + alg_name + "_" + model.hash_id
        Path(path).mkdir(parents=True, exist_ok=True)

        if existing_entry.shape[0] > 0:
            fitness = existing_entry['fitness'][0]
            print(f"Model for this solution already exists")
            return fitness

        else:
            # TODO Find a more optimal way
            """Punishing bad decisions"""
            if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
                RMSE = int(9e10)
                AUC = 0.0
            else:
                experiment = DNNAEExperiment(model, config['exp_params'], config['model_params']['n_features'])
                config['trainer_params']['max_epochs'] = model.num_epochs
                tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                              name=str(self.iteration) + "_" + alg_name + "_" + model.hash_id)

                runner = Trainer(logger=tb_logger,
                                 enable_progress_bar=False,
                                 accelerator="gpu",
                                 devices=1,
                                 auto_select_gpus=True,
                                 callbacks=[
                                     LearningRateMonitor(),
                                     ModelCheckpoint(save_top_k=1,
                                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                                     monitor="val_loss",
                                                     save_last=True),
                                     early_stop_callback,
                                 ],
                                 # strategy=DDPPlugin(find_unused_parameters=False),
                                 **config['trainer_params'])

                print(f"======= Training {config['model_params']['name']} =======")
                print(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                runner.fit(experiment, datamodule=datamodule)
                print(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')

                # TODO Find a more optimal way
                # Known problem: https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/5
                if math.isnan(experiment.test_RMSE.item()):
                    RMSE = int(9e10)
                    AUC = experiment.AUC
                else:
                    RMSE = experiment.test_RMSE.item()
                    AUC = experiment.AUC

            complexity = (model.num_epochs ** 2) + (model.num_layers * 100) + (model.bottleneck_size * 10)
            fitness = (RMSE * 1000) + (complexity / 100)

            print(tabulate([[RMSE, AUC, complexity, fitness]], headers=["RMSE", "AUC", "Complexity", "Fitness"],
                           tablefmt="pretty"))
            conn.post_entries(model, fitness, solution, RMSE, AUC, complexity, alg_name, self.iteration)
            torch.save(model.state_dict(), path + f"/model.pt")

            return fitness


if __name__ == '__main__':
    print(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    print(f"RUN UUID: {RUN_UUID}")
    """
    Dimensionality:
    y1: topology shape,
    y2: number of neurons per layer,
    y3: number of layers,
    y4: activation function
    y5: number of epochs,
    y6: learning rate
    y7: optimizer algorithm.
    """
    DIMENSIONALITY = 7

    runner = ExtendedRunner(
        config['logging_params']['save_dir'],
        dimension=DIMENSIONALITY,
        max_evals=1000,
        runs=1,
        algorithms=[
            ParticleSwarmAlgorithm(),
            DifferentialEvolution(),
            FireflyAlgorithm(),
            SelfAdaptiveDifferentialEvolution(),
            GeneticAlgorithm()
        ],
        problems=[
            DNNAEArchitecture(DIMENSIONALITY)
        ]
    )

    print("=====================================SEARCH STARTED==============================================")
    final_solutions = runner.run(export='json', verbose=True)
    print("=====================================SEARCH COMPLETED============================================")

    best_solution, best_algorithm = conn.best_results()
    best_model = Autoencoder(best_solution, **config)
    model_file = config['logging_params']['save_dir'] + f"{best_algorithm}_{best_model.hash_id}.pt"
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    torch.save(best_model.state_dict(), model_file)
    print(f"Best model saved to: {model_file}")
    print(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
