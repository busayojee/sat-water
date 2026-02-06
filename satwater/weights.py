"""
Created on Fri Jan 16 16:21:44 2026

@author: Busayo Alabi
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from huggingface_hub import hf_hub_download


class WeightsError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeightsRef:
    model_key: str
    repo_id: str
    revision: str
    hf_root: str
    weights_file_in_repo: str
    expected_sha256: str
    input_shape: str | None = None


def _sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(repo_id, revision, hf_root="weights", cache_dir=None):
    """
    Download and parse manifest from HF.
    """
    manifest_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=f"{hf_root.strip('/')}/manifest.json",
        revision=revision,
        cache_dir=cache_dir,
    )
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_weights_ref(
    model_key, repo_id, revision="main", hf_root="weights", cache_dir=None
):
    """
    Resolve a model_key into the exact file and expected sha256 from manifest.
    """
    manifest = _load_manifest(
        repo_id=repo_id, revision=revision, hf_root=hf_root, cache_dir=cache_dir
    )

    if "models" not in manifest or not isinstance(manifest["models"], dict):
        raise WeightsError("Invalid manifest.json: missing 'models' mapping")

    models = manifest["models"]
    if model_key not in models:
        available = ", ".join(sorted(models.keys()))
        raise WeightsError(f"Unknown model_key='{model_key}'. Available: {available}")

    entry = models[model_key]
    try:
        weights_file = entry["weights_file"]
        sha = entry["sha256"]
        input_shape = entry.get("input_shape")
    except KeyError as e:
        raise WeightsError(
            f"Invalid manifest entry for '{model_key}': missing {e!s}"
        ) from e
    weights_file_in_repo = f"{hf_root.strip('/')}/{weights_file}".replace("//", "/")

    return WeightsRef(
        model_key=model_key,
        repo_id=repo_id,
        revision=revision,
        hf_root=hf_root.strip("/"),
        weights_file_in_repo=weights_file_in_repo,
        expected_sha256=sha,
        input_shape=input_shape,
    )


def download_weights(
    model_key,
    repo_id,
    revision="main",
    hf_root="weights",
    cache_dir=None,
    verify=True,
    retry_on_mismatch=True,
):
    """
    Download weights for a given model_key, verify SHA256, and return the local path.
    """
    ref = resolve_weights_ref(
        model_key=model_key,
        repo_id=repo_id,
        revision=revision,
        hf_root=hf_root,
        cache_dir=cache_dir,
    )

    def _download(force):
        rel = ref.weights_file_in_repo
        return hf_hub_download(
            repo_id=ref.repo_id,
            repo_type="model",
            filename=rel,
            revision=ref.revision,
            cache_dir=cache_dir,
            force_download=force,
        )

    local_path = _download(force=False)

    if not verify:
        return local_path

    actual = _sha256_file(local_path)
    if actual == ref.expected_sha256:
        return local_path

    if retry_on_mismatch:
        # Corrupted cache or partial download. Force another new download once.
        local_path = _download(force=True)
        actual = _sha256_file(local_path)
        if actual == ref.expected_sha256:
            return local_path

    raise WeightsError(
        "SHA256 mismatch for downloaded weights.\n"
        f"model_key: {ref.model_key}\n"
        f"repo_id: {ref.repo_id}\n"
        f"revision: {ref.revision}\n"
        f"file: {ref.weights_file_in_repo}\n"
        f"expected: {ref.expected_sha256}\n"
        f"actual: {actual}\n"
        "Tip: If you are offline or behind a proxy, downloads may be partial or may not work."
    )


DEFAULT_WEIGHTS_REPO = os.environ.get(
    "SATWATER_WEIGHTS_REPO", "busayojee/sat-water-weights"
)
DEFAULT_WEIGHTS_REV = os.environ.get("SATWATER_WEIGHTS_REV", "main")


def get_weights_path(
    model_key,
    repo_id=DEFAULT_WEIGHTS_REPO,
    revision=DEFAULT_WEIGHTS_REV,
    hf_root="weights",
    cache_dir=None,
):
    # fpr inference
    return download_weights(
        model_key=model_key,
        repo_id=repo_id,
        revision=revision,
        hf_root=hf_root,
        cache_dir=cache_dir,
        verify=True,
        retry_on_mismatch=True,
    )
