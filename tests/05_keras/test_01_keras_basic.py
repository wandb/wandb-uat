#!/usr/bin/env python

import argparse
import os
import types
from typing import List, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback


def get_model() -> keras.Model:
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


def get_dataset(data_size: int) -> Tuple[np.ndarray, np.ndarray]:
    images = np.random.rand(data_size, 28, 28, 1)
    labels = tf.keras.utils.to_categorical(
        np.random.randint(0, high=10, size=(data_size,)), 10
    )

    return images, labels


def main(args: Union[argparse.Namespace, types.SimpleNamespace]) -> None:
    run = wandb.init(name=__file__, sync_tensorboard=args.tensorboard)
    run_path = run.path

    callbacks: List = []
    if args.tensorboard:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=run.dir, update_freq="batch"
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

    if not os.environ.get("WB_UAT_SKIP_CHECK"):
        check(run_path, tensorboard=args.tensorboard)


def check(run_path: str, tensorboard: bool = False) -> None:
    api = wandb.Api()
    api_run = api.run(run_path)
    assert api_run.summary["loss"] >= 0
    assert api_run.state == "finished"
    if tensorboard:
        assert api_run.summary["train/epoch_loss"] >= 0
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
        "-e",
        "--epochs",
        default=2,
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
