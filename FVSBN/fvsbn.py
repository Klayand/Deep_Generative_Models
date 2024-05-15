# Fully Visible Sigmoid Belief Network
# Author: Zikai Zhou

# The FVBN is an autoregressive model composed of a collection of linear models. Each
# linear model tries to model p(x_i|x_{j<i}). The FVBN can be viewed as a special case of
# MADE [2] with no hidden layers and one set of masks applied in raster-scan order.
#
# References (used throughout code):
#     [1]: https://www.semanticscholar.org/paper/Connectionist-Learning-of-Belief-Networks-Neal/a120c05ad7cd4ce2eb8fb9697e16c7c4877208a5
#     [2]: https://arxiv.org/pdf/1502.03509.pdf


# Ensemble Version: Parameters 10 X 10 = 100 >> (10 - 1)*10 / 2 = 45
# Iteration Version: Parameters (10 - 1)*10 / 2 = 45 << 10 X 10 = 100

import random
from typing import List

import torch
import torch.nn as nn
import torch.distributions as dist
from matplotlib import pyplot as plt


class FVSBN(nn.Module):  # Ensemble version
    def __init__(self, input_dim):
        super(FVSBN, self).__init__()
        # X --> [0, 1], so input_dim = output_dim
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim)

        # Fix weights as 0 for indices >= row for each row.
        for row, weights in enumerate(self.linear.weight.data):
            weights[row: ].data.fill_(0)

    # For a given input x, obtain the mean vectors describing the
    # Bernoulli distributions for each dimension, and each sample
    def mean_vectors(self, x):
        return torch.sigmoid(self.linear(x))

    # Forward pass to compute the log-likelihoods for each input separately
    def forward(self, x):
        bernoulli_mean = self.mean_vectors(x)
        log_bernoulli_mean = torch.log(bernoulli_mean)
        log_likelihoods = x * log_bernoulli_mean + (1 - x) * (1 - log_bernoulli_mean)
        return torch.sum(log_likelihoods, dim=1)

    # Do not update weights for indices >= row for each row
    def zero_grad_for_extra_weights(self):
        for row, grads in enumerate(self.linear.weight.grad):
            grads[row: ] = 0

    # Sample
    @torch.no_grad()
    def sample(self, num_samples):
        samples = torch.zeros(num_samples, self.input_dim)
        for sample_num in range(num_samples):
            sample = torch.zeros(self.input_dim)
            for dim in range(self.input_dim):
                weights = self.linear.weight.data[dim]
                bias = self.linear.bias.data[dim]
                bernoulli_mean_dim = torch.sigmoid(sample.matmul(weights) + bias)
                distribution = dist.bernoulli.Bernoulli(probs=bernoulli_mean_dim)
                sample[dim] = distribution.sample()

            samples[sample_num] = sample

        return samples


class FVSBN_v2(nn.Module):  # Iteration Version
    def __init__(self, input_dim, output_dim=1):
        super(FVSBN_v2, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

    # For a given input x, obtain the mean vectors describing the
    # Bernoulli distributions for each dimension, and each sample
    def mean_vectors(self, x):
        return torch.sigmoid(self.linear(x))

    # Forward pass to compute the log-likelihoods for each input separately
    def forward(self, x):
        x = torch.cat(*x, dim=0)

        bernoulli_mean = self.mean_vectors(x)
        log_bernoulli_mean = torch.log(bernoulli_mean)
        log_likelihoods = x * log_bernoulli_mean + (1 - x) * (1 - log_bernoulli_mean)
        return torch.sum(log_likelihoods, dim=1)


class FVSBN_v3(nn.Module):  # Iteration Version v2
    def __init__(self, num_layers, input_dim, output_dim=1):
        super(FVSBN_v3, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.criterion = nn.BCELoss()

        # We use in_features=1 and always pass an input of 0 for the first linear
        self.block = nn.ModuleList([nn.Linear(max(1, i), 1) for i in range(num_layers)])

    # For a given input x, obtain the mean vectors describing the
    # Bernoulli distributions for each dimension, and each sample
    def mean_vectors(self, x):
        return torch.sigmoid(self.linear(x))

    # Forward pass to compute the log-likelihoods for each input separately
    # Or we can design a loss to implement this forward function
    # Here we just maximize the log-likelihood to get the optimized parameters
    def forward(self, x):

        # pass 0 to the first linear layer
        output = torch.sigmoid(self.block[0](torch.zeros(x.shape[0], 1)))
        log_likelihoods = (output * torch.log(output) + (1 - output) * (1 - torch.log(output))).squeeze()

        for i in range(1, self.input_dim):
            output = torch.sigmoid(self.block[i](x[:, :i]))
            log_bernoulli_mean = torch.log(output)
            log_likelihoods += (output * log_bernoulli_mean + (1 - output) * (1 - log_bernoulli_mean)).squeeze()

        return log_likelihoods / self.num_layers

        # --------- with BCE loss function optimization ----------
        # loss = 0
        # for i in range(0, self.input_dim - 1):
        #     output = torch.sigmoid(self.block[i](x[:, :i+1]))  # [128, 1]
        #     loss += self.criterion(output, x[:, i].reshape(x.shape[0], -1))
        # return loss / self.num_layers

    @torch.no_grad()
    def sample(self, num_samples=1):
        samples = torch.zeros(num_samples, self.input_dim)  # 28*28*1 = 784

        for sample_num in range(num_samples):
            sample = torch.zeros(self.input_dim)

            random_value = random.random()
            bernoulli_dist = dist.Bernoulli(torch.tensor(random_value))
            x0 = bernoulli_dist.sample().reshape(1, -1)

            input_list = [x0]

            for i in range(1, self.input_dim):
                output = torch.sigmoid(self.block[i](torch.cat(input_list, dim=1) if len(input_list) > 1 else x0))
                distribution = dist.bernoulli.Bernoulli(probs=output)
                sample[i] = distribution.sample()
                input_list.append(sample[i].reshape(1, -1))

            samples[sample_num] = sample

        samples = samples.reshape(num_samples, 1, 28, 28)  # mnist [1, 28, 28]

        return samples


if __name__ == '__main__':
    input = torch.randint(0, 2, size=(1, 784), dtype=torch.float32)
    model = FVSBN_v3(num_layers=784, input_dim=784, output_dim=1)
    log_likelihood = model(input)
    samples = model.sample(num_samples=2)

    for sample in samples:
        plt.imshow(sample.squeeze()*255, cmap='gray')
        plt.show()



