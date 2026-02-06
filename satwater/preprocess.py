"""
Created on Wed Nov 17 10:48:32 2023

@author: Busayo Alabi
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import tensorflow as tf
except Exception as e:
    raise ImportError(
        "TensorFlow is required for sat-water inference/training.\n\n"
        "Install TensorFlow first, then reinstall sat-water.\n"
        "Recommended:\n"
        "  Linux/Windows: pip install 'tensorflow'\n"
        "  Apple Silicon: pip install 'tensorflow-macos' 'tensorflow-metal'\n\n"
        "If you are using segmentation-models with TF legacy Keras:\n"
        "  pip install tf-keras segmentation-models\n"
    ) from e

from skimage import color


class Preprocess:
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    @staticmethod
    def load(mask_path, mask_folder, image_folder, channels):
        mask = tf.io.read_file(mask_path)
        mask = tf.dtypes.cast(tf.image.decode_jpeg(mask, channels=1), tf.float32)
        img_path = tf.strings.regex_replace(mask_path, mask_folder, image_folder)
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=channels)
        return {"image": Preprocess.normalization_layer(image), "mask": mask}

    @staticmethod
    def make_single_class(sample):
        mask = sample["mask"]
        sample["mask"] = tf.where(mask >= 0.5, 1, 0)
        return sample

    @staticmethod
    def resize(data, size):
        image = tf.image.resize(data["image"], size[:2])
        mask = tf.image.resize(data["mask"], size[:2])
        return image, mask

    def resize_test(data, size):
        image = tf.image.resize(data, size)
        return image

    @staticmethod
    def data_load(
        dataset, mask_folder, image_folder, split, shape, batch_size, channels=0
    ):
        dataset = [dataset]
        ds = []
        for f in dataset:
            mask_path = list(Path(f + mask_folder).glob("**/*"))
            for i in mask_path:
                ds.append(str(i))
        size = len(ds)
        random.shuffle(ds)
        ds = tf.data.Dataset.from_tensor_slices(ds)
        val_size, test_size = split
        train_size = int((1 - (val_size + test_size)) * size)
        val_size = int(val_size * size)
        test_size = int(test_size * size)
        train_ds = ds.take(train_size)
        rest_ds = ds.skip(train_size)
        val_ds = rest_ds.take(val_size)
        test_ds = rest_ds.skip(val_size)

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        print(f"The Train Dataset contains {len(train_ds)} images.")
        print(f"The Validation Dataset contains {len(val_ds)} images.")
        print(f"The Test Dataset contains {len(test_ds)} images.")

        train_ds = train_ds.map(
            lambda x: Preprocess.load(x, mask_folder, image_folder, channels)
        )
        train_ds = train_ds.map(Preprocess.make_single_class)
        train_ds = train_ds.map(
            lambda x: Preprocess.resize(x, shape), num_parallel_calls=AUTOTUNE
        )
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

        val_ds = val_ds.map(
            lambda x: Preprocess.load(x, mask_folder, image_folder, channels)
        )
        val_ds = val_ds.map(Preprocess.make_single_class)
        val_ds = val_ds.map(
            lambda x: Preprocess.resize(x, shape), num_parallel_calls=AUTOTUNE
        )
        val_ds = val_ds.batch(batch_size)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

        test_ds = test_ds.map(
            lambda x: Preprocess.load(x, mask_folder, image_folder, channels)
        )
        test_ds = test_ds.map(Preprocess.make_single_class)
        test_ds = test_ds.map(
            lambda x: Preprocess.resize(x, shape), num_parallel_calls=AUTOTUNE
        )
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

        return (train_ds, val_ds, test_ds)

    @staticmethod
    def label_overlay(img, mask, fname=None):
        plt.figure()

        mask = np.squeeze(mask, axis=-1)

        overlay = color.label2rgb(
            mask,
            img,
            colors=[(0, 255, 0)],
            alpha=0.05,
            bg_label=0,
            bg_color=None,
            image_alpha=1,
            saturation=1,
            kind="overlay",
        )
        plt.figure(figsize=(6, 6))

        plt.imshow(overlay)
        plt.axis("off")

        if fname:
            plt.savefig(fname)
            plt.clf()
            plt.close()

        plt.show()

    @staticmethod
    def plot_image(train_ds, n=1, fname=None):
        for image, mask in (
            train_ds.unbatch().shuffle(buffer_size=len(train_ds)).take(1)
        ):
            image = (image - tf.reduce_min(image)) / (
                tf.reduce_max(image) - tf.reduce_min(image)
            )

            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(image)
            plt.subplot(122)
            plt.imshow(mask)
            plt.show()
            Preprocess.label_overlay(image, mask, fname)
