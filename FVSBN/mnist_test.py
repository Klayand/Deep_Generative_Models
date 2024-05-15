import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from fvsbn import FVSBN_v3


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    CHANNELS_IMG = 1
    NUM_EPOCHES = 5

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)]
            )
        ]
    )

    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FVSBN_v3(num_layers=784, input_dim=784)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(NUM_EPOCHES):
        for batch_idx, (real, _) in enumerate(loader):

            model.zero_grad()

            # Compute log-likelihoods per sample.
            log_likelihoods = model(real.view(BATCH_SIZE, -1))

            # Negative mean over all samples, because we're minimizing with SGD instead of maximizing.
            negative_mean_log_likehoods = -torch.mean(log_likelihoods)
            # loss = torch.mean(log_likelihoods)

            # Compute gradients.
            negative_mean_log_likehoods.backward()
            # loss.backward()

            # Update weights.
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHES}] Batch {batch_idx}/{len(loader)} NLL: {negative_mean_log_likehoods.item():.4f}"
                )
                # print(
                #     f"Epoch [{epoch}/{NUM_EPOCHES}] Batch {batch_idx}/{len(loader)} loss: {loss.item():.4f}"
                # )

                samples = model.sample(num_samples=2)
                for sample in samples:
                    plt.imshow(sample.squeeze() * 255, cmap='gray')
                    plt.show()