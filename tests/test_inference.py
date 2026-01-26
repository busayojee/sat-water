from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from satwater.inference import Inference
from satwater.preprocess import Preprocess


def _write_dummy_jpeg(path: Path, size=(600, 600)) -> None:
    """Create a dummy RGB JPEG on disk."""
    arr = (np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


def _save_dummy_segmentation_model(
    path: Path, input_shape=(128, 128, 3), classes=2
) -> None:
    """Small softmax conv model saved as .h5"""
    inp = tf.keras.Input(shape=input_shape)
    out = tf.keras.layers.Conv2D(classes, 1, activation="softmax")(inp)
    model = tf.keras.Model(inp, out)
    model.save(path)


def _save_dummy_flexible_model(path: Path, classes=2) -> None:
    """Model that accepts variable H/W: (None, None, 3). Useful for both 128 and 512 in dict inference."""
    inp = tf.keras.Input(shape=(None, None, 3))
    out = tf.keras.layers.Conv2D(classes, 1, activation="softmax")(inp)
    model = tf.keras.Model(inp, out)
    model.save(path)


def test_predict_inst_single_unet_smoke(tmp_path: Path, monkeypatch):
    """
    This covers single model instance image inference runs
    """
    # Import inside test so pytest can skip in case TF isn't availble
    import matplotlib.pyplot as plt

    # prevent GUI/file writes during tests
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    model_path = tmp_path / "dummy_unet.keras"
    _save_dummy_segmentation_model(model_path, input_shape=(128, 128, 3), classes=2)

    img_path = tmp_path / "img.jpg"
    _write_dummy_jpeg(img_path)

    infer = Inference(model=str(model_path), name="unet")
    infer.predict_inst(str(img_path), fname="test_unet")


def test_predict_inst_dict_models_smoke(tmp_path: Path, monkeypatch):
    """
    This covers all models,
    where image2 (512x512) is used and label_overlay is called.
    """
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(Preprocess, "label_overlay", lambda *args, **kwargs: None)

    model_path = tmp_path / "dummy_flexible.keras"
    _save_dummy_flexible_model(model_path, classes=2)

    models = {
        "unet": str(model_path),
        "resnet34": str(model_path),
        "resnet34(2)": str(model_path),
    }

    img_path = tmp_path / "img.jpg"
    _write_dummy_jpeg(img_path)

    infer = Inference(model=models)
    infer.predict_inst(str(img_path), fname="test_multi")
