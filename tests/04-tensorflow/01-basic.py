import argparse
from pathlib import Path
import sys

if not sys.modules.get("tensorflow"):
    sys.exit(0)

import numpy as np
import tensorflow as tf
import wandb
from tensorflow import keras


class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(32, 3, activation="relu")
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(128, activation="relu")
        self.d2 = keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


@tf.function
def train_step(
    model, images, labels, loss_object, optimizer, train_loss, train_accuracy
):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def get_dataset(size: int, batch: int):
    images = np.random.rand(size, 28, 28, 1)
    labels = keras.utils.to_categorical(np.random.randint(0, high=10, size=(size,)), 10)
    return (
        tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(size).batch(batch)
    )


def main(args):
    # Create an instance of the model
    model = MyModel()

    dataset = get_dataset(args.data_size, args.batch)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name="train_accuracy"
    )

    run = wandb.init(sync_tensorboard=args.tensorboard)

    if args.tensorboard:
        log_dir = Path().cwd() / "wandb" / "runs"  # TODO clear this directory
        writer = tf.summary.create_file_writer(str(log_dir))

    for i, (images, labels) in enumerate(dataset):
        train_step(
            model,
            images,
            labels,
            loss_object,
            optimizer,
            train_loss,
            train_accuracy,
        )
        print(
            f"Step:\t{i}\tLoss:\t{train_loss.result().numpy():.3}"
            f"\tAccuracy:\t{train_accuracy.result().numpy():.3}"
        )
        run.log({"mean_loss": train_loss.result().numpy()})
        run.log({"acc": train_accuracy.result().numpy()})

        if args.tensorboard:
            with writer.as_default():
                tf.summary.scalar("loss", train_loss.result(), i)
                tf.summary.scalar("accuracy", train_accuracy.result(), step=i)

    run.finish()
    if args.tensorboard:
        writer.close()


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
