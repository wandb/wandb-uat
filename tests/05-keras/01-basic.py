#!/usr/bin/env python

import argparse
import sys
from pathlib import Path
from typing import List

import keras
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback


def get_model():
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )


def get_dataset(data_size):
    images = np.random.rand(data_size, 28, 28, 1)
    labels = tf.keras.utils.to_categorical(
        np.random.randint(0, high=10, size=(data_size,)), 10
    )

    return (images, labels)


def main(args):

    run = wandb.init(sync_tensorboard=args.tensorboard)

    callbacks: List = []
    if args.tensorboard:
        log_dir = Path().cwd() / "wandb" / "runs"  # TODO clear this directory
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, update_freq="batch"
            )
        )

    callbacks.append(WandbCallback(monitor="accuracy"))

    model = get_model()
    dataset = get_dataset(args.data_size)

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.fit(
        *dataset,
        batch_size=args.batch,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    run.finish()


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
        "-e",
        "--epochs",
        default=3,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        action="store_true",
        help="Add TensorBoard syncing",
    )
    args = parser.parse_args()

    main(args)
