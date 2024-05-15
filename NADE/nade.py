# Implementation of Neural Autoregressive Distribution Estimation
# Author: Zikai Zhou

# NADE can be viewed as a one hidden layer autoencoder masked to satisfy the
# autoregressive property. This masking allows NADE to act as a generative model
# by explicitly estimating p(X) as a factor of conditional probabilities, i.e,
# P(X) = \prod_i^D p(X_i|X_{j<i}), where X is a feature vector and D is the
# dimensionality of X.

# [1]: https://arxiv.org/abs/1605.02226

import torch
import torch.nn as nn
import torch.distributions as dist
from matplotlib import pyplot as plt


class NADE(nn.Module):
    """The Neural Autoregressive Distribution Estimation"""
    def __init__(self, input_dim, hidden_dim):
        """

        :param input_dim: The dimension of the input
        :param hidden_dim: The dimension of the hidden layer. NADE only
            supports one hidden layer.
        """
        super(NADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Here we dive into the shape of NADE, B -> batch size, N -> input dim, d -> hidden dim
        # X -> (B, N), x_i -> (1, N)
        # W -> (d, N), for each dimension of each x_i, we get h_ii -> (d, 1)
        # so h_i -> (d, N), H -> (B, d, N)
        # for each h_ii, we have v_ii -> (d, 1), o_ii -> (1, 1)
        # o_i -> (1, N), O -> (B, N)
        # parameters: N*d + d + n*d + n + d = 2*n*d + 2d + n
        self._in_W = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self._in_b = nn.Parameter(torch.zeros(hidden_dim, ))
        self._h_W = nn.Parameter(torch.zeros(input_dim, hidden_dim))
        self._h_b = nn.Parameter(torch.zeros(input_dim, ))

        # initialization
        nn.init.kaiming_normal_(self._in_W)
        nn.init.kaiming_normal_(self._h_W)

    def _forward(self, x, num_samples=2):
        """
            Computes the forward pass and samples a new output
        :return:
            (p_hat, x_hat) where p_hat is the probability distribution over
            dimensions and x_hat is sampled from p_hat
            
            x = torch.tensor([[1], [2], [3]])
            x.size()
            torch.Size([3, 1])
            x.expand(3, 4)
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
            x.expand(-1, 4)   # -1 means not changing the size of that dimension
            tensor([[ 1,  1,  1,  1],
                    [ 2,  2,  2,  2],
                    [ 3,  3,  3,  3]])
        """
        p_hat, x_hat = [], []
        batch_size = num_samples if x is None else x.shape[0]

        # Only the bias is used to compute the first hidden unit, so we must
        # replicate it to account for the batch size
        a = self._in_b.expand(batch_size, -1)  # h1 (B, d)

        for i in range(self.input_dim):
            # compute p1, x1
            h = torch.relu(a)
            p_i = torch.sigmoid(h @ self._h_W[i : i + 1, :].t() + self._h_b[i : i + 1])  # [5, 1]
            p_hat.append(p_i)

            # sample 'x' at dimension 'i'
            x_i = x[:, i : i + 1]
            x_i = torch.where(x_i < 0, dist.Bernoulli(probs=p_i).sample(), x_i)  # [5, 1]
            x_hat.append(x_i)

            # We do not need to add self._in_b[i] when computing the other hidden
            # units since it was already added when computing the first hidden unit
            a = a + x_i @ self._in_W[:, i : i+1].t()

        return torch.cat(p_hat, dim=1), torch.cat(x_hat, dim=1) if x_hat else []

    def forward(self, x):
        """
            Computes the forward pass
        :param x: either a tensor of vectors with shape (n, input_dim) or images with
            shape (n, 1, h, w) here h*w = input_dim

        :return: the result of the forward pass
        """
        return self._forward(x)[0]

    @torch.no_grad()
    def sample(self, num_samples=2):
        samples = torch.ones((num_samples, self.input_dim)) * -1
        samples = self._forward(x=samples, num_samples=num_samples)[1]

        for sample in samples:
            plt.imshow(sample.squeeze().reshape(28, 28) * 255, cmap='gray')
            plt.show()


if __name__ == '__main__':
    input = torch.randn((5, 784))
    model = NADE(input_dim=784, hidden_dim=20)
    p_hat, x_hat = model._forward(input)
    print(x_hat.shape)

    model.sample(num_samples=2)
