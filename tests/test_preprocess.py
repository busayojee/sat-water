from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from satwater.preprocess import Preprocess

pytestmark = pytest.mark.tf


def _write_dummy_pair(root: Path, idx: int, size=(64, 64)) -> None:
    masks = root / "Masks"
    images = root / "Images"
    masks.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)

    img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(images / f"{idx}.jpg", format="JPEG")

    m = (np.random.rand(*size) > 0.5).astype(np.uint8) * 255
    Image.fromarray(m, mode="L").save(masks / f"{idx}.jpg", format="JPEG")


def test_load_returns_image_and_mask(tmp_path: Path):
    _write_dummy_pair(tmp_path, 0)

    sample = Preprocess.load(
        str(tmp_path / "Masks" / "0.jpg"),
        "/Masks",
        "/Images",
        channels=3,
    )

    assert set(sample.keys()) == {"image", "mask"}
    assert sample["image"].shape[-1] == 3
    assert sample["mask"].shape[-1] == 1


def test_make_single_class_outputs_binary_mask(tmp_path: Path):
    _write_dummy_pair(tmp_path, 0)

    sample = Preprocess.load(
        str(tmp_path / "Masks" / "0.jpg"),
        "/Masks",
        "/Images",
        channels=3,
    )

    out = Preprocess.make_single_class(sample)
    uniq = np.unique(out["mask"].numpy())
    assert set(uniq.tolist()).issubset({0.0, 1.0})


def test_resize_returns_expected_shapes(tmp_path: Path):
    _write_dummy_pair(tmp_path, 0)

    sample = Preprocess.load(
        str(tmp_path / "Masks" / "0.jpg"),
        "/Masks",
        "/Images",
        channels=3,
    )
    sample = Preprocess.make_single_class(sample)

    image, mask = Preprocess.resize(sample, (32, 32, 3))
    assert tuple(image.shape)[:2] == (32, 32)
    assert tuple(mask.shape)[:2] == (32, 32)


def test_data_load_splits_and_batches(tmp_path: Path):
    for i in range(20):
        _write_dummy_pair(tmp_path, i)

    train_ds, val_ds, test_ds = Preprocess.data_load(
        dataset=str(tmp_path),
        mask_folder="/Masks",
        image_folder="/Images",
        split=[0.2, 0.1],
        shape=(32, 32, 3),
        batch_size=4,
        channels=3,
    )

    x, y = next(iter(train_ds))
    assert x.ndim == 4
    assert y.ndim == 4
    assert x.shape[1:3] == (32, 32)
    assert y.shape[1:3] == (32, 32)
