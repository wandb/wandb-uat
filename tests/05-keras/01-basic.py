import keras
import numpy as np
import wandb
from wandb.keras import WandbCallback


def main():

    run = wandb.init()
    model = keras.Sequential(
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
    x_train = np.random.rand(60000, 28, 28, 1)
    y_train = keras.utils.to_categorical(
        np.random.randint(0, high=10, size=(60000,)), 10
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=3,
        validation_split=0.1,
        callbacks=[WandbCallback()],
    )

    run.finish()


if __name__ == "__main__":
    main()
