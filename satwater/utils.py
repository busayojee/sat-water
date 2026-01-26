from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def set_seed(seed=42):
    """Make runs more reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path):
    """Create a directory if it doesn't exist and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_shape(shape):
    """
    Parse shape string like '128,128,3' into (128, 128, 3).
    """
    parts = [int(x.strip()) for x in shape.split(",")]
    if len(parts) != 3:
        raise ValueError("shape must be 'H,W,C' e.g. '128,128,3'")
    return parts[0], parts[1], parts[2]


def parse_models_arg(models_arg):
    """
    Parse "unet=path,resnet34=path,resnet34(2)=path"  dict.
    """
    out: dict[str, str] = {}
    for item in models_arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("models must be comma-separated key=path pairs")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out
