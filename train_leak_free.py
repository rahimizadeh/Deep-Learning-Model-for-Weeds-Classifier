"""Leak-free baseline training pipeline for the weed image dataset.

The important design rule is: split ORIGINAL files first, then apply stochastic
augmentation only to the training dataset. Validation and test data remain
untouched except for resizing/rescaling.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_samples(dataset_dir: Path):
    paths, labels = [], []
    class_dirs = sorted(
        [p for p in dataset_dir.iterdir() if p.is_dir()],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    class_names = [p.name for p in class_dirs]
    class_to_index = {name: index for index, name in enumerate(class_names)}

    for class_dir in class_dirs:
        label = class_to_index[class_dir.name]
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(path))
                labels.append(label)

    if not paths:
        raise ValueError(f"No images found in {dataset_dir}")

    return np.asarray(paths), np.asarray(labels), class_names


def split_samples(paths, labels, seed: int = SEED):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temp_labels,
    )
    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def decode_image(path, label, image_size):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(paths, labels, image_size, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda path, label: decode_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int, image_size):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="train_only_augmentation",
    )

    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Weeds Dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", default="weed_classifier.keras")
    args = parser.parse_args()

    set_seed()
    image_size = (args.image_size, args.image_size)
    paths, labels, class_names = collect_samples(Path(args.data))
    train, val, test = split_samples(paths, labels)

    print(f"Classes: {class_names}")
    print(f"Train: {len(train[0])} | Validation: {len(val[0])} | Test: {len(test[0])}")
    print("Augmentation is applied inside the model and is active only during training.")

    train_ds = make_dataset(*train, image_size, args.batch_size, training=True)
    val_ds = make_dataset(*val, image_size, args.batch_size, training=False)
    test_ds = make_dataset(*test, image_size, args.batch_size, training=False)

    model = build_model(len(class_names), image_size)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Final untouched test accuracy: {test_accuracy:.4f}")
    print(f"Final untouched test loss: {test_loss:.4f}")
    model.save(args.output)


if __name__ == "__main__":
    main()
