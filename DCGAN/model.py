import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_dim):
        super(Discriminator, self).__init__()

        # input: N x C x 64 x 64
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_dim, features_dim * 2, 4, 2, 1),
            self._block(features_dim * 2, features_dim * 4, 4, 2, 1),
            self._block(features_dim * 4, features_dim * 8, 4, 2, 1),
            nn.Conv2d(features_dim*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_dim * 16, 4, 1, 0),
            self._block(features_dim * 16, features_dim * 8, 4, 2, 1),
            self._block(features_dim * 8, features_dim * 4, 4, 2, 1),
            self._block(features_dim * 4, features_dim * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_dim * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100

    x = torch.randn((N, in_channels, H, W))

    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    initialize_weights(gen)
    assert gen(z).shape == (N, in_channels, H, W)

    print("Successful!")
