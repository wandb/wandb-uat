#!/usr/bin/env python

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import wandb


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, size: int = 6000) -> None:
        self.img_labels = torch.randint(10, (size,))
        self.imgs = torch.randn(size, 1, 28, 28)

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.imgs[idx]
        label = self.img_labels[idx]
        return image, label


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    dataloader: DataLoader,
) -> None:

    run = wandb.init()

    for data in dataloader:
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        run.log({"loss": loss})

    run.finish()


def main() -> None:
    # Construct our model by instantiating the class defined above
    model = Net()

    dataloader = DataLoader(CustomDataset(), batch_size=64, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model, optimizer, loss_fn, dataloader)


if __name__ == "__main__":
    main()
