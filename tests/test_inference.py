from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from satwater.inference import Inference

pytestmark = pytest.mark.tf
# TODO: IMPROVE TEST SCRIPTS TO INCLUDE NEW FUNCTIONS


def _write_dummy_jpeg(path: Path, size=(600, 600)) -> None:
    arr = (np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


def _save_dummy_segmentation_model(path: Path, input_shape=(128, 128, 3), classes=2):
    inp = tf.keras.Input(shape=input_shape)
    out = tf.keras.layers.Conv2D(classes, 1, activation="softmax")(inp)
    model = tf.keras.Model(inp, out)
    model.save(path)


def _save_dummy_flexible_model(path: Path, classes=2):
    inp = tf.keras.Input(shape=(None, None, 3))
    out = tf.keras.layers.Conv2D(classes, 1, activation="softmax")(inp)
    model = tf.keras.Model(inp, out)
    model.save(path)


def test_predict_single_local_model_smoke(tmp_path: Path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "imsave", lambda *args, **kwargs: None)

    model_path = tmp_path / "dummy_unet.keras"
    _save_dummy_segmentation_model(model_path, input_shape=(128, 128, 3), classes=2)

    img_path = tmp_path / "img.jpg"
    _write_dummy_jpeg(img_path)

    infer = Inference(model=str(model_path), name="unet")
    res = infer.predict(
        str(img_path), return_overlay=True, save=False, show=False, fname="test"
    )

    assert "unet" in res.masks
    assert res.masks["unet"].ndim == 3
    assert res.masks["unet"].shape[-1] == 1

    assert "unet" in res.overlays
    assert res.overlays["unet"].ndim == 3
    assert res.overlays["unet"].shape[-1] == 3


def test_predict_multi_local_models_smoke(tmp_path: Path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "imsave", lambda *args, **kwargs: None)

    model_path = tmp_path / "dummy_flexible.keras"
    _save_dummy_flexible_model(model_path, classes=2)

    models = {
        "unet": str(model_path),
        "resnet34_256": str(model_path),
        "resnet34_512": str(model_path),
    }

    img_path = tmp_path / "img.jpg"
    _write_dummy_jpeg(img_path)

    infer = Inference(model=models)
    res = infer.predict(
        str(img_path), return_overlay=True, save=False, show=False, fname="multi"
    )

    assert set(res.masks.keys()) == {"unet", "resnet34_256", "resnet34_512"}
    assert set(res.overlays.keys()) == {"unet", "resnet34_256", "resnet34_512"}

    for k in res.masks:
        assert res.masks[k].ndim == 3
        assert res.masks[k].shape[-1] == 1

    for k in res.overlays:
        assert res.overlays[k].ndim == 3
        assert res.overlays[k].shape[-1] == 3
