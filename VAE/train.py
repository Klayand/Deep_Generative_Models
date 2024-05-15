import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torchvision.transforms as transforms

from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from model import VariationalAutoencoder


# Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e3

# Dataset Loading
dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoencoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss(reduction='sum')

# Start Training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))

    for i, (x, _) in loop:
        # Forward pass
        x = x.to(DEVICE).view(x.shape[0], -1)
        x_reconstructed, mu, sigma = model(x)

        # Compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma)

        # Backprop
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


model = model.to('cpu')


def inference(digit, num_example=1):
    images = []
    idx = 0

    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digits = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digits.append((mu, sigma))

    mu, sigma = encodings_digits[digit]
    for example in range(num_example):
        epilson = torch.randn_like(sigma)
        z = mu + sigma * epilson

        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"Generated_{digit}_ex{example}.png")


if __name__ == '__main__':
    for idx in range(10):
        inference(idx, num_example=1)


















