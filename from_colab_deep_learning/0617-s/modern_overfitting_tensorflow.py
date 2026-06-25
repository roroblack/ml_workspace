"""
Modern overfitting-prevention training code for the 0617 TensorFlow/Keras lab.

Key fixes over the notebook:
- actual label smoothing via one-hot labels + CategoricalCrossentropy
- EarlyStopping with min_delta, warm-up, explicit mode
- ModelCheckpoint and ReduceLROnPlateau
- optional EMA weights through AdamW(use_ema=True) + SwapEMAWeights
- compact CNN instead of pure MLP
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


SEED = 42
NUM_CLASSES = 10
BEST_MODEL_PATH = Path("best_fashion_cnn.keras")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data():
    (x_train_all, y_train_all), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train_all = x_train_all.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train_all = np.expand_dims(x_train_all, -1)
    x_test = np.expand_dims(x_test, -1)

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(x_train_all))
    train_idx = indices[:5000]
    val_idx = indices[5000:10000]

    x_train = x_train_all[train_idx]
    y_train = keras.utils.to_categorical(y_train_all[train_idx], NUM_CLASSES)
    x_val = x_train_all[val_idx]
    y_val = keras.utils.to_categorical(y_train_all[val_idx], NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model() -> keras.Model:
    data_augmentation = keras.Sequential(
        [
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomRotation(factor=0.08),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            layers.RandomContrast(factor=0.1),
        ],
        name="data_augmentation",
    )

    inputs = keras.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs)

    for filters, dropout in [(32, 0.15), (64, 0.25)]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="regularized_fashion_cnn")


def make_callbacks() -> list[keras.callbacks.Callback]:
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            min_delta=1e-4,
            start_from_epoch=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_delta=1e-4,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    if hasattr(keras.callbacks, "SwapEMAWeights"):
        callbacks.insert(0, keras.callbacks.SwapEMAWeights(swap_on_epoch=True))
    return callbacks


def train_modern(epochs: int = 50, batch_size: int = 128):
    set_seed()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    model = build_model()

    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_ema=True,
        ema_momentum=0.995,
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=make_callbacks(),
        verbose=1,
    )

    best_model = keras.models.load_model(BEST_MODEL_PATH) if BEST_MODEL_PATH.exists() else model
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test | loss={test_loss:.4f} acc={test_acc:.4f}")
    return best_model, history


if __name__ == "__main__":
    train_modern()
