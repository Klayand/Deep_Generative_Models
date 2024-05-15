import torch
import torch.nn as nn
import torch.nn.functional as F


# input image -> hidden vector -> mean, std -> Parameterization Trick -> decode -> output
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, z_dim=20):
        super().__init__()

        # encoder
        self.img2hid = nn.Linear(input_dim, hidden_dim)
        self.hid2mu= nn.Linear(hidden_dim, z_dim)
        self.hid2sigma = nn.Linear(hidden_dim, z_dim)

        # decoder
        self.z2hid = nn.Linear(z_dim, hidden_dim)
        self.hid2img = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img2hid(x))
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z2hid(z))
        return torch.sigmoid(self.hid2img(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma * epsilon
        x_reconstructed = self.decode(z_new)

        return x_reconstructed, mu, sigma


if __name__ == '__main__':
    x = torch.randn(4, 28*28)  # mnist --> (28, 28, 1) = 784
    vae = VariationalAutoencoder(input_dim=784)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)
