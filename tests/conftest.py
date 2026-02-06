from __future__ import annotations

from pathlib import Path


def _has_tensorflow():
    try:

        return True
    except Exception:
        return False


def pytest_ignore_collect(collection_path: Path, config):
    p = str(collection_path)
    if p.endswith(("test_inference.py", "test_models.py", "test_preprocess.py")):
        return not _has_tensorflow()
    return False
