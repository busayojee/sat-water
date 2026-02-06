"""
Created on Fri Jan 16 19:43:12 2026

@author: Busayo Alabi
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from satwater.weights import get_weights_path

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

try:
    import segmentation_models as sm
except Exception:
    sm = None


class BuilderError(RuntimeError):
    pass


def build_custom_unet(input_shape=(128, 128, 3), n_classes=2):
    """
    Pretrained U-Net builder matching exported weights.
    """

    def conv_block(x, filters: int):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

    def encoder_block(x, filters: int):
        c = conv_block(x, filters)
        p = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(c)
        return p, c

    def decoder_block(x, skip, filters: int):
        x = tf.keras.layers.Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding="same"
        )(x)
        x = tf.keras.layers.Concatenate(axis=-1)([skip, x])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

    inputs = tf.keras.layers.Input(shape=input_shape)

    p0, c0 = encoder_block(inputs, 32)
    p1, c1 = encoder_block(p0, 64)
    p2, c2 = encoder_block(p1, 128)
    p3, c3 = encoder_block(p2, 256)
    p4, c4 = encoder_block(p3, 512)

    center = conv_block(p4, 1024)

    d4 = decoder_block(center, c4, 512)
    d3 = decoder_block(d4, c3, 256)
    d2 = decoder_block(d3, c2, 128)
    d1 = decoder_block(d2, c1, 64)
    d0 = decoder_block(d1, c0, 32)

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation="softmax")(d0)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="satwater_unet")


@dataclass(frozen=True)
class PretrainedModel:
    """
    Container returned by `load_pretrained()`.
    args
    model: tf.keras.Model ready for inference
    preprocess_func: optional preprocessing for backbones, for exampkle Resnet34
    input_shape: expected (H, W, C)
    model_key: key saved in manifest : "unet" | "resnet34_256" | "resnet34_512"
    """

    model: tf.keras.Model
    preprocess_func: Callable | None
    input_shape: tuple[int, int, int]
    model_key: str


MODEL_SPECS = {
    "unet": (128, 128, 3),
    "resnet34_256": (256, 256, 3),
    "resnet34_512": (512, 512, 3),
}


def _require_segmentation_models():
    if sm is None:
        raise ImportError(
            "segmentation-models is required for ResNet34 pretrained models.\n"
            "Install it with:\n"
            "  pip install segmentation-models tf-keras\n"
        )


def build_resnet34_unet(input_shape, n_classes=2):
    _require_segmentation_models()
    model = sm.Unet(
        "resnet34",
        classes=n_classes,
        activation="softmax",
        encoder_weights=None,
        input_shape=input_shape,
    )
    preprocess_func = sm.get_preprocessing("resnet34")
    return model, preprocess_func


def load_pretrained(
    model_key, repo_id=None, revision="main", n_classes=2, hf_root="weights"
):
    """
    Download and verify pretrained weights from Hugging Face, build the correct architecture,
    load weights, and then return a ready model.

    Returns the pretrained model
    """
    if model_key not in MODEL_SPECS:
        raise BuilderError(
            f"Unknown model_key='{model_key}'. Expected the following: {', '.join(MODEL_SPECS.keys())}"
        )
    input_shape = MODEL_SPECS[model_key]

    weights_path = get_weights_path(
        model_key=model_key,
        repo_id=repo_id
        or os.environ.get("SATWATER_WEIGHTS_REPO", "busayojee/sat-water-weights"),
        revision=revision,
        hf_root=hf_root,
    )

    preprocess_func = None
    if model_key == "unet":
        raise NotImplementedError("Unet not ready yet")
    else:
        model, preprocess_func = build_resnet34_unet(
            input_shape=input_shape, n_classes=n_classes
        )

    model.load_weights(weights_path)
    return PretrainedModel(
        model=model,
        preprocess_func=preprocess_func,
        input_shape=input_shape,
        model_key=model_key,
    )
