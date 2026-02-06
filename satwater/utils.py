from __future__ import annotations

import os
import random
from pathlib import Path

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


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_shape(shape):
    parts = [int(x.strip()) for x in shape.split(",")]
    if len(parts) != 3:
        raise ValueError("shape must be 'H,W,C' e.g. '128,128,3'")
    return parts[0], parts[1], parts[2]


def parse_models_arg(models_arg):
    out = {}
    for item in models_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("models must be comma-separated key=path pairs")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out
