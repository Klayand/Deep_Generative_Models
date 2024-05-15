import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from nade import NADE


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 200
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

    dataset = datasets.MNIST(root='../MADE/data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NADE(input_dim=784, hidden_dim=500)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(NUM_EPOCHES):
        for batch_idx, (real, _) in enumerate(loader):

            # for visualization
            real_image = real[0].squeeze()

            model.zero_grad()

            # Compute log-likelihoods per sample.
            p_hats = model(real.view(BATCH_SIZE, -1))

            loss = F.binary_cross_entropy_with_logits(p_hats, real.view(BATCH_SIZE, -1), reduction="none").sum(dim=1).mean()

            loss.backward()
            # loss.backward()

            # Update weights.
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHES}] Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}"
                )
                real_sample = real[0]
                plt.imshow(real_sample.reshape(28, 28) * 255, cmap='gray')
                plt.show()

                samples = model.sample(num_samples=1)