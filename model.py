import cnnmodel as cnn
import unetmodel as unet
import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_size: int) -> None:
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.init_size = img_size // 4

        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.model = unet.UNetAuto()

    def forward(self, X):
        return self.model(X)
    
class Discriminator(nn.Module):
    def __init__(self, latent_dim: int, img_size: int) -> None:
        super(Discriminator, self).__init__()

        self.latent_dim = latent_dim
        self.init_size = img_size // 8
        self.model = cnn()

    def forward(self, X):
        out = self.model(X)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# https://github.com/eriklindernoren/PyTorch-GAN#auxiliary-classifier-gan
# Studying examples