import hashlib
import random
import time

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from tabulate import tabulate

from models.base import BaseAutoencoder
from models.types_ import *


class Autoencoder(BaseAutoencoder, nn.Module):

    def __init__(self, solution, **kwargs) -> None:
        super(Autoencoder, self).__init__()

        n_features = kwargs['model_params']['n_features']
        seq_len = kwargs['model_params']['seq_len']
        batch_size = kwargs['data_params']['batch_size']

        self.id = str(int(time.time())).strip()
        self.dataset_shape = [n_features, seq_len]
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.topology_shape = self.map_shape(solution[0])
        self.layer_step = self.map_layer_step(solution[1], self.dataset_shape)
        self.num_layers = self.map_num_layers(solution[2], self.layer_step, self.dataset_shape)
        self.activation = self.map_activation(solution[3])
        self.num_epochs = self.map_num_epochs(solution[4])
        self.learning_rate = self.map_learning_rate(solution[5])

        self.bottleneck_size = 0
        self.seq_len = seq_len
        self.n_features = n_features
        self.batch_size = batch_size

        self.generate_autoencoder(self.topology_shape,
                                  self.num_layers,
                                  self.dataset_shape,
                                  self.layer_step)

        self.optimizer = self.map_optimizer(solution[6])
        self.get_hash()
        outputs = []

        outputs.append([self.hash_id,
                        self.topology_shape,
                        self.layer_step,
                        self.num_layers,
                        self.activation_name,
                        self.num_epochs,
                        self.learning_rate,
                        self.optimizer_name,
                        self.bottleneck_size,
                        self.encoding_layers,
                        self.decoding_layers])

        print(tabulate(outputs, headers=["ID",
                                         "Shape (y1)",
                                         "Layer step (y2)",
                                         "Layers (y3)",
                                         "Activation func. (y4)",
                                         "Epochs (y5)",
                                         "Learning rate (y6)",
                                         "Optimizer (y7)",
                                         "Bottleneck size",
                                         "Encoder",
                                         "Decoder", ], tablefmt="pretty"))

    def get_hash(self):

        self.hash_id = hashlib.sha1(str(str(self.topology_shape) +
                                        str(self.layer_step) +
                                        str(self.num_layers) +
                                        str(self.activation_name) +
                                        str(self.num_epochs) +
                                        str(self.learning_rate) +
                                        str(self.optimizer_name) +
                                        str(self.bottleneck_size) +
                                        str(self.encoding_layers) +
                                        str(self.decoding_layers)).encode('utf-8')).hexdigest()

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        encoded = x.view(x.size(0), -1)

        for layer in self.encoding_layers:
            result = layer(encoded)
            encoded = self.activation(result)

        return encoded

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        decoded = z

        for layer in self.decoding_layers:
            result = layer(decoded)
            decoded = self.activation(result)

        """Flipping back to original shape"""
        reconstructed = decoded
        return reconstructed

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """Flipping shape of tensors"""

        z = self.encode(input)
        reconstructed = self.decode(z)

        return [reconstructed, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the AE loss function.
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss

        details = {'loss': loss, 'Reconstruction_Loss': recons_loss.detach()}
        return details

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.bottleneck_size)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        reconstructed, input = self.forward(x)
        return [reconstructed, input]

    def map_shape(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.5])
        inds = np.digitize(gene, bins)

        if inds[0] - 1 == 0:
            return "SYMMETRICAL"

        elif inds[0] - 1 == 1:
            return "A-SYMMETRICAL"

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def map_layer_step(self, gene, dataset_shape):
        gene = np.array([gene])
        bins = []
        value = 1 / dataset_shape[1]
        step = value
        for col in range(0, dataset_shape[1]):
            bins.append(step)
            step += value
        bins[-1] = 1.01
        inds = np.digitize(gene, bins)
        return inds[0]

    def map_num_layers(self, gene, layer_step, dataset_shape):
        if layer_step == 0:
            max_layers = dataset_shape[1]
            return max_layers

        else:
            max_layers = round(dataset_shape[1] / layer_step)

        if max_layers == 1:
            return 1

        else:
            gene = np.array([gene])

            bins = []
            value = 1 / max_layers
            step = value
            for col in range(0, max_layers):
                bins.append(step)
                step += value
            bins[-1] = 1.01
            inds = np.digitize(gene, bins)

            return int(inds[0])

    def map_activation(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.125, 0.25, 0.375, 0.500, 0.625, 0.750, 0.875, 1.01])
        inds = np.digitize(gene, bins)

        if inds[0] - 1 == 0:
            self.activation_name = "ELU"
            return F.elu

        elif inds[0] - 1 == 1:
            self.activation_name = "RELU"
            return F.relu

        elif inds[0] - 1 == 2:
            self.activation_name = "Leaky RELU"
            return F.leaky_relu

        elif inds[0] - 1 == 3:
            self.activation_name = "RRELU"
            return F.rrelu

        elif inds[0] - 1 == 4:
            self.activation_name = "SELU"
            return F.selu

        elif inds[0] - 1 == 5:
            self.activation_name = "CELU"
            return F.celu

        elif inds[0] - 1 == 6:
            self.activation_name = "GELU"
            return F.gelu

        elif inds[0] - 1 == 7:
            self.activation_name = "TANH"
            return torch.tanh

        else:

            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

    def map_num_epochs(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.7, 0.8, 0.9, 1.01])
        inds = np.digitize(gene, bins)

        return int(inds[0]) * 10 + 100

    def map_learning_rate(self, gene):
        gene = np.array([gene])
        bins = []
        value = 1 / 1000
        step = value
        for col in range(0, 1000):
            bins.append(step)
            step += value
        bins[-1] = 1.01
        inds = np.digitize(gene, bins)
        lr = np.array(bins)[inds[0]]

        return round(lr, 2)

    def generate_autoencoder(self, shape, layers, dataset_shape, layer_step):
        if shape == "SYMMETRICAL":

            i = dataset_shape[1]
            z = dataset_shape[1] - layer_step

            while layers != 0:
                """Minimum depth reached"""
                if z < 1:
                    self.encoding_layers.append(nn.Linear(in_features=i, out_features=z + 1))
                    self.decoding_layers.insert(0, nn.Linear(in_features=z + 1, out_features=i))
                    self.bottleneck_size = z + 1
                    break

                self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))
                self.decoding_layers.insert(0, nn.Linear(in_features=z, out_features=i))
                i = i - layer_step
                z = z - layer_step
                layers = layers - 1

            if len(self.encoding_layers) == 0:
                self.bottleneck_size = 0
            else:
                self.bottleneck_size = self.encoding_layers[-1].out_features

        elif shape == "A-SYMMETRICAL":
            i = dataset_shape[1]
            z = dataset_shape[1] - layer_step

            if layers == 1 or layers == 2:
                self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))
                self.decoding_layers.insert(0, nn.Linear(in_features=z, out_features=i))

            if layers >= 3:
                layers_encoder = random.randint(1, layers)
                layers_decoder = layers - layers_encoder

                encoder_counter = layers_encoder
                decoder_counter = layers_decoder

                if layers_decoder == 0:
                    layers_encoder = layers_encoder - 1
                    layers_decoder = 1

                    encoder_counter = layers_encoder
                    decoder_counter = layers_decoder

                while encoder_counter != 0:

                    if z < 1:
                        self.encoding_layers.append(nn.Linear(in_features=i, out_features=z + 1))
                        self.bottleneck_size = z + 1
                        break

                    self.encoding_layers.append(nn.Linear(in_features=i, out_features=z))

                    i = i - layer_step
                    z = z - layer_step
                    encoder_counter = encoder_counter - 1

                while decoder_counter != 0:

                    if layers_decoder == 1:
                        self.decoding_layers.insert(0, nn.Linear(in_features=i, out_features=dataset_shape[1]))
                        break

                    layer_step = int((dataset_shape[1] - i) / decoder_counter)  # Make more complex logic
                    last_i = i
                    i = i + layer_step
                    z = z + layer_step
                    decoder_counter = decoder_counter - 1

                    self.decoding_layers.append(nn.Linear(in_features=last_i, out_features=i))

            if len(self.encoding_layers) == 0:
                self.bottleneck_size = 0
            else:
                self.bottleneck_size = self.encoding_layers[-1].out_features

    def map_optimizer(self, gene):
        gene = np.array([gene])
        bins = np.array([0.0, 0.167, 0.334, 0.50, 0.667, 0.834, 1.01])
        inds = np.digitize(gene, bins)

        """When AE does not have any layers"""
        if len(list(self.parameters())) == 0:
            self.optimizer_name = "Empty"
            return None

        if inds[0] - 1 == 0:
            self.optimizer_name = "Adam"
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 1:
            self.optimizer_name = "Adagrad"
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 2:
            self.optimizer_name = "SGD"
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 3:
            self.optimizer_name = "RAdam"
            return torch.optim.RAdam(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 4:
            self.optimizer_name = "ASGD"
            return torch.optim.ASGD(self.parameters(), lr=self.learning_rate)

        elif inds[0] - 1 == 5:
            self.optimizer_name = "RPROP"
            return torch.optim.Rprop(self.parameters(), lr=self.learning_rate)

        else:
            raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")
