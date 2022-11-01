from models.base import BaseVAE
from models.dnn_ae import Autoencoder
from models.rnn_vae import RNNVAE
from models.vanilla_vae import VanillaVAE

# Aliases
VAE = VanillaVAE
RNNVAE = RNNVAE

vae_models = {'VanillaVAE':VAE,
              'RNNVAE':RNNVAE,
              'Autoencoder': Autoencoder}

__all__ = ["RNNVAE", "VanillaVAE", "Autoencoder", "BaseVAE", "vae_models"]
__import__("pkg_resources").declare_namespace(__name__)


