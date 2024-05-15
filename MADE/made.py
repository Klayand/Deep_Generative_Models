"""
Implementation of Masked Autoencoder Distribution Estimator (MADE) [1].
From : CS236

MADE is an extension of NADE [2] which allows using arbitrarily deep fully
connected networks as the distribution estimator. More specifically, MADE is a
deep, fully-connected autoencoder masked to respect the autoregressive property.
For any ordering of the input features, MADE only uses features j<i to predict
feature i. This property allows MADE to be used as a generative model by
specifically modelling P(X) = \prod_i^D p(X_i|X_{j<i}) where X is an input
feature and D is the dimensionality of X.

[1]: https://arxiv.org/abs/1502.03509
[2]: https://arxiv.org/abs/1605.02226
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from matplotlib import pyplot as plt


class MaskedLinear(nn.Linear):
    """
        A Linear layer with masks that turn off some of the layer`s weights.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        """
            self.register_buffer() --> fixed parameters
            self.register_parameter() --> unfixed parameters
        """
        # Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
        self.register_buffer('mask', torch.ones((out_features, in_features)))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, x):
        # element-wise product --> do not change the shape
        self.weight.data *= self.mask
        return super().forward(x)


class MADE(nn.Module):
    """
        The Masked Autoencoder Distribution Estimator (MADE) model.
    """
    def __init__(self, input_dim, hidden_dim: List or None = None, n_mask=1):
        """
        Initialize a new MADE model.

        :param input_dim: The dimensionality of the input.
        :param hidden_dim: A list containing the number of hidden units for each hidden layer.
        :param n_mask: The total number of distinct masks to use during training/eval
        """
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.dims = [self.input_dim] + (hidden_dim or []) + [self.input_dim]
        self.n_mask = n_mask

        self.mask_seed = 0

        # define the model architecture
        layers = []
        for i in range(len(self.dims) - 1):
            in_dim, out_dim = self.dims[i], self.dims[i + 1]
            layers.append(MaskedLinear(in_dim, out_dim))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers[:-1], nn.Sigmoid())  # remove the last ReLU()

        # for layer in self.net.modules():
        #     if isinstance(layer, MaskedLinear):
        #         nn.init.xavier_uniform_(layer.weight)

    def _sample_masks(self):
        """Samples a new set of autoregressive masks.

        Only 'self.n_masks' distinct sets of masks are sampled after which the mask
        sets are rotated through in the order in which they were sampled. In
        principle, it's possible to generate the masks once and cache them. However,
        this can lead to memory issues for large 'self.n_masks' or models many
        parameters. Finally, sampling the masks is not that computationally
        expensive.

        :return:
            A tuple of (masks, ordering). Ordering refers to the ordering of the
            outputs since MADE is order agnostic.
        """

        # make sure the masks for each layer are different
        rng = np.random.RandomState(seed=self.mask_seed % self.n_mask)
        self.mask_seed += 1

        # sample connectivity patterns
        conn = [rng.permutation(self.input_dim)]  # [(784, )]
        for i, dim in enumerate(self.dims[1:-1]):  # intermediate layer
            # Note: The dimensions in the paper are 1-indexed whereas
            # arrays in python are 0-indexed. Implementation adjusted accordingly.
            low = 0 if i == 0 else np.min(conn[i-1])
            high = self.input_dim - 1

            conn.append(rng.randint(low, high, size=dim))

        conn.append(np.copy(conn[0]))  # (784, ) (100, ) (200, ) (784, )

        # create masks
        # except for last layer
        # "None" will create a new dimension
        # for example conn[0] -> (784, ) => conn[0][None, :] -> (1, 784)
        #             conn[1] -> (100, ) => conn[1][:, None] -> (100, 1)
        masks = [
            conn[i - 1][None, :] <= conn[i][:, None] for i in range(1, len(conn) - 1)
        ]

        # for last layer
        masks.append(conn[-2][None, :] < conn[-1][:, None])

        # [(100, 784), (200, 100), (784, 200)]
        # `cause need to do matrix multiplication

        return [torch.from_numpy(mask.astype(np.uint8)) for mask in masks], conn[-1]

    def _forward(self, x, masks):
        layers = [
            layer for layer in self.net.modules() if isinstance(layer, MaskedLinear)
        ]

        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)

        return self.net(x)

    def forward(self, x):
        """
            Compute the forward pass
        :param x:
            Either a tensor of vectors with shape (n, input_dim) or images with shape
            (n, 1, h, w) where h * w = input_dim.
        :return:
            The result of the forward pass.
        """

        masks, _ = self._sample_masks()
        return self._forward(x, masks)

    @torch.no_grad()
    def _sample(self, x):
        """
                    x = np.array([3, 1, 2])
                    >> > np.argsort(x)
                    array([1, 2, 0])
                """
        masks, ordering = self._sample_masks()
        ordering = np.argsort(ordering)  # get the corresponding index

        # sequence generation by the order
        for dim in ordering:
            out = self._forward(x, masks)[:, dim]
            out = dist.Bernoulli(probs=out).sample()

            # yes, out; otherwise, x[:, dim]
            # this is very important, it updates the value
            x[:, dim] = torch.where(x[:, dim] < 0, out, x[:, dim])

        return x

    @torch.no_grad()
    def sample(self, num_samples):
        samples = torch.ones((num_samples, self.input_dim)) * -1
        outputs = self._sample(samples)

        for sample in outputs:
            plt.imshow(sample.squeeze().reshape(28, 28) * 255, cmap='gray')
            plt.show()


if __name__ == "__main__":
    x = torch.randn(5, 784)
    model = MADE(input_dim=784, hidden_dim=[8000])
    model.sample(num_samples=5)










