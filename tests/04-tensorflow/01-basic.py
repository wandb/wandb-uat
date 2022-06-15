#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb


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

    loss = train_loss(loss)
    acc = train_accuracy(labels, predictions)
    # print(f"loss: {loss}, acc: {acc}")
    return loss, acc


def train():
    # Create an instance of the model
    model = MyModel()

    x_train = np.random.rand(60000, 28, 28, 1)
    y_train = keras.utils.to_categorical(
        np.random.randint(0, high=10, size=(60000,)), 10
    )
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
    )

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

    run = wandb.init()

    for images, labels in train_ds:
        loss, acc = train_step(
            model, images, labels, loss_object, optimizer, train_loss, train_accuracy
        )
        run.log({"mean_loss": loss.numpy()})
        run.log({"acc": acc.numpy()})

    run.finish()


if __name__ == "__main__":
    train()
