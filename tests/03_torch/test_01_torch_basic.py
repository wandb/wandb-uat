#!/usr/bin/env python

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, size: int) -> None:
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
    sync_tensorboard: bool,
) -> str:

    run: wandb.sdk.wandb_run.Run = wandb.init(
        sync_tensorboard=sync_tensorboard
    )  # type: ignore
    run_path = run.path

    if sync_tensorboard:
        writer = SummaryWriter(run.dir)

    for i, data in enumerate(dataloader):
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        run.log({"loss": loss})
        print(f"Step:\t{i}\tLoss:\t{loss:.3}")

        if sync_tensorboard:
            writer.add_scalar("training loss", loss, i)

    if sync_tensorboard:
        writer.close()
    run.finish()

    return run_path


def main(args) -> None:
    # Construct our model by instantiating the class defined above
    model = Net()

    dataloader = DataLoader(
        CustomDataset(args.data_size), batch_size=args.batch, shuffle=True
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    run_path = train(model, optimizer, loss_fn, dataloader, args.tensorboard)
    if not os.environ.get("WB_UAT_SKIP_CHECK"):
        check(run_path, tensorboard=args.tensorboard)


def check(run_path, tensorboard=False):
    api = wandb.Api()
    api_run = api.run(run_path)
    assert api_run.summary["loss"] >= 0
    assert api_run.state == "finished"
    if tensorboard:
        assert api_run.summary["training loss"] >= 0
        assert api_run.summary["global_step"] > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-size",
        default=60_000,
        type=int,
        help="Total dataset size",
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=128,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        action="store_true",
        help="Add TensorBoard syncing",
    )
    args = parser.parse_args()

    main(args)
