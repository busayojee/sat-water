"""
Created on Wed Nov 18 13:22:57 2023

@author: Busayo Alabi
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

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

from satwater.builders import load_pretrained
from satwater.preprocess import Preprocess

try:
    from PIL import Image
except Exception:
    Image = None

ArrayLikeImage = Union[str, np.ndarray, tf.Tensor, "Image.Image"]


@dataclass(frozen=True)
class SegmentationResult:
    """
    Result from a segmentation call.
    """

    masks: dict[str, np.ndarray]
    overlays: dict[str, np.ndarray]
    base_image: np.ndarray


class Inference:
    """
    Supports both pretrained model keys and local saved model paths:
    """

    pretrained_keys = {"unet", "resnet34_256", "resnet34_512"}

    def __init__(
        self, model="unet", *, name=None, repo_id=None, revision="main", save_dir=None
    ):
        self.repo_id = repo_id or os.environ.get(
            "SATWATER_WEIGHTS_REPO", "busayojee/sat-water-weights"
        )
        self.revision = revision
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.name = name
        self.preprocess_func = None
        self.preprocess_funcs = {}

        if isinstance(model, dict):
            self.model = {}
            for key, spec in model.items():
                m, pp = self._load_one(spec, key_hint=key)
                self.model[key] = m
                self.preprocess_funcs[key] = pp
            return

        if isinstance(model, str):
            if model in self.pretrained_keys:
                m, pp = self._load_pretrained(model)
                self.model = m
                self.name = model
                self.preprocess_func = pp
                return

            if self._looks_like_file(model):
                self.model = tf.keras.models.load_model(model, compile=False)
                self.name = name or "local"
                return

            raise ValueError(
                "model must be a pretrained key (unet/resnet34_256/resnet34_512), "
                "a local path to a saved model, or a dict of them."
            )

        raise TypeError("model must be a str or dict[str, str]")

    @staticmethod
    def _looks_like_file(s):
        return ("/" in s) or s.endswith(".keras") or s.endswith(".h5")

    def _load_pretrained(self, key):
        pm = load_pretrained(
            model_key=key, repo_id=self.repo_id, revision=self.revision
        )
        return pm.model, pm.preprocess_func

    def _load_one(self, spec, key_hint=None):
        if spec in self.pretrained_keys:
            return self._load_pretrained(spec)
        if self._looks_like_file(spec):
            return tf.keras.models.load_model(spec, compile=False), None
        if key_hint in self.pretrained_keys:
            return self._load_pretrained(key_hint)
        raise ValueError(
            f"Unknown model spec '{spec}'. Use pretrained key or local model path."
        )

    @staticmethod
    def _to_tf_image(x):
        if isinstance(x, str):
            raw = tf.io.read_file(x)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            img = tf.cast(img, tf.float32) / 255.0
            return img

        if Image is not None and isinstance(x, Image.Image):
            arr = np.asarray(x.convert("RGB"), dtype=np.float32) / 255.0
            return tf.convert_to_tensor(arr, dtype=tf.float32)

        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = arr.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            return tf.convert_to_tensor(arr, dtype=tf.float32)

        if isinstance(x, tf.Tensor):
            t = tf.cast(x, tf.float32)
            if t.shape.rank == 2:
                t = tf.stack([t, t, t], axis=-1)
            if t.shape.rank == 3 and t.shape[-1] == 4:
                t = t[..., :3]
            tmax = tf.reduce_max(t)
            t = tf.cond(tmax > 1.5, lambda: t / 255.0, lambda: t)
            return t

        raise TypeError(
            "Unsupported image type. Use path, PIL.Image, numpy array, or tf.Tensor."
        )

    @staticmethod
    def _resize(img, size):
        return tf.image.resize(img, size=size, method="bilinear")

    @staticmethod
    def _mask_to_uint8(mask):
        m = np.squeeze(mask)
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        return m

    @staticmethod
    def overlay_mask(image, mask, alpha=0.55, color=(0.0, 0.4, 1.0)):
        img = image
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0.0, 1.0)

        m = np.squeeze(mask)
        water = (m > 0).astype(np.float32)
        water3 = np.repeat(water[:, :, None], 3, axis=2)

        col = np.array(color, dtype=np.float32)[None, None, :]
        overlay = img * (1.0 - alpha * water3) + col * (alpha * water3)
        return (np.clip(overlay, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _maybe_save(self, arr, fname):
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        out = self.save_dir / fname
        plt.imsave(out.as_posix(), arr)

    def predict(
        self, image, *, return_overlay=True, save=False, show=False, fname="prediction"
    ):
        raw_img = self._to_tf_image(image)
        model_img = Preprocess.normalization_layer(raw_img)
        base_img = np.clip(raw_img.numpy(), 0.0, 1.0).astype(np.float32)

        masks = {}
        overlays = {}

        if isinstance(self.model, dict):
            for key, model in self.model.items():
                mask_np = self._predict_for_key(key, model, model_img)
                masks[key] = mask_np
                if return_overlay:
                    ov = self.overlay_mask(
                        self._image_for_key(key, raw_img).numpy(), mask_np
                    )
                    overlays[key] = ov
                    if save and self.save_dir is not None:
                        self._maybe_save(ov, f"{fname}_{key}_overlay.png")
                        self._maybe_save(np.squeeze(mask_np), f"{fname}_{key}_mask.png")
            if show:
                self._plot_results(base_img, masks, overlays, title=fname)
            return SegmentationResult(
                masks=masks, overlays=overlays, base_image=base_img
            )

        key = self.name or "model"
        mask_np = self._predict_for_key(key, self.model, model_img)
        masks[key] = mask_np

        if return_overlay:
            ov = self.overlay_mask(self._image_for_key(key, raw_img).numpy(), mask_np)
            overlays[key] = ov
            if save and self.save_dir is not None:
                self._maybe_save(ov, f"{fname}_{key}_overlay.png")
                self._maybe_save(np.squeeze(mask_np), f"{fname}_{key}_mask.png")

        if show:
            self._plot_results(base_img, masks, overlays, title=fname)

        return SegmentationResult(masks=masks, overlays=overlays, base_image=base_img)

    def _image_for_key(self, key, img):
        if key == "unet":
            return self._resize(img, (128, 128))
        if key == "resnet34_256":
            return self._resize(img, (256, 256))
        if key == "resnet34_512":
            return self._resize(img, (512, 512))
        return self._resize(img, (128, 128))

    def _predict_for_key(self, key, model, img):
        x = self._image_for_key(key, img)

        if isinstance(self.model, dict):
            pp = self.preprocess_funcs.get(key)
        else:
            pp = self.preprocess_func

        if pp is not None:
            x = pp(x)

        x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-8)

        pred = model.predict(x[tf.newaxis, ...], verbose=0)
        pred = tf.argmax(pred, axis=-1)
        pred = tf.expand_dims(pred, axis=-1)[0, :, :, :]
        pred_np = pred.numpy().astype(np.int32)

        if key == "resnet34_512":
            pred_np = 1 - pred_np
        return pred_np

    @staticmethod
    def _plot_results(base_img, masks, overlays, title="result"):
        keys = list(masks.keys())
        n = len(keys)
        cols = 3 if overlays else 2
        plt.figure(figsize=(5 * cols, 4 * n))

        for i, k in enumerate(keys):
            r = i * cols

            plt.subplot(n, cols, r + 1)
            plt.title("Image")
            plt.imshow(base_img)
            plt.axis("off")

            plt.subplot(n, cols, r + 2)
            plt.title(f"Mask: {k}")
            plt.imshow(np.squeeze(masks[k]))
            plt.axis("off")

            if overlays:
                plt.subplot(n, cols, r + 3)
                plt.title(f"Overlay: {k}")
                plt.imshow(overlays[k])
                plt.axis("off")

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def segment_image(
    image,
    *,
    model="unet",
    repo_id=None,
    revision="main",
    return_overlay=True,
    save=False,
    save_dir=None,
    show=False,
    fname="prediction",
):
    infer = Inference(
        model=model, repo_id=repo_id, revision=revision, save_dir=save_dir
    )
    return infer.predict(
        image, return_overlay=return_overlay, save=save, show=show, fname=fname
    )
