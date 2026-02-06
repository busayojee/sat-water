"""
Created on Fri Jan 15 14:16:32 2026

@author: Busayo Alabi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

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

from huggingface_hub import HfApi

print("TF imported from:", getattr(tf, "__file__", None))
print("TF version:", getattr(tf, "__version__", None))
print("Has tf.keras?", hasattr(tf, "keras"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    path: Path
    input_shape: str | None = None


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model_path(p):
    path = Path(p).expanduser()
    if path.exists():
        return path.resolve()
    for ext in (".keras", ".h5"):
        candidate = Path(str(path) + ext).expanduser()
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Model file not found: {p} (also tried .keras/.h5)")


def export_weights(model_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(model_path), compile=False)

    weights_path = out_dir / f"{model_path.stem}.weights.h5"
    model.save_weights(str(weights_path))
    return weights_path


def write_sha_file(out_dir, weights_files: list[Path]):
    sha_path = out_dir / "SHA256SUMS.txt"
    lines = []
    for wf in weights_files:
        lines.append(f"{sha256_file(wf)}  {wf.name}")
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha_path


def write_manifest(out_dir, repo_id, hf_root, entries: dict):
    manifest = {
        "repo_id": repo_id,
        "hf_root": hf_root.strip("/"),
        "models": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def upload_to_hf(repo_id, local_dir, path_in_repo, token, private: bool):
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload sat-water weights to {path_in_repo}",
    )


def parse_model_args(model_args: list[str]):
    """
    Accept repeated args like:
      --model unet=~/.../model1
      --model resnet34_256=~/.../model2
      --model resnet34_512=~/.../model3
    shape metadata (optional):
      --model unet=~/.../model1@128,128,3
    """
    specs = []
    for item in model_args:
        if "=" not in item:
            raise ValueError("--model must be KEY=PATH (optionally KEY=PATH@H,W,C)")
        key, rest = item.split("=", 1)
        key = key.strip()
        shape = None
        if "@" in rest:
            path_str, shape = rest.split("@", 1)
            path_str = path_str.strip()
            shape = shape.strip()
        else:
            path_str = rest.strip()
        specs.append(
            ModelSpec(key=key, path=resolve_model_path(path_str), input_shape=shape)
        )
    return specs


def main():
    p = argparse.ArgumentParser(
        description="Export Keras weights and upload to Hugging Face (multi-model)."
    )
    p.add_argument("--repo-id", required=True, help="e.g. busayojee/sat-water-weights")
    p.add_argument(
        "--hf-root",
        default="weights",
        help="Root folder inside HF repo to store everything (default: weights/)",
    )
    p.add_argument(
        "--out-dir",
        default="dist/weights",
        help="Local directory to write generated artifacts",
    )
    p.add_argument("--private", action="store_true", help="Create HF repo as private")
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: KEY=PATH or KEY=PATH@H,W,C. Repeat for multiple models.",
    )
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Export HF_TOKEN before running.")
    specs = parse_model_args(args.model)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export each model into its own subfolder under out_dir/<key>/
    manifest_entries = {}
    weights_files = []
    for spec in specs:
        model_out = out_dir / spec.key
        model_out.mkdir(parents=True, exist_ok=True)
        weights_path = export_weights(spec.path, model_out)
        weights_files.append(weights_path)
        sha = sha256_file(weights_path)
        manifest_entries[spec.key] = {
            "source_model": str(spec.path),
            "weights_file": f"{spec.key}/{weights_path.name}",
            "sha256": sha,
            "bytes": weights_path.stat().st_size,
            "input_shape": spec.input_shape,
        }

    sha_file = write_sha_file(out_dir, weights_files)
    manifest_path = write_manifest(
        out_dir, args.repo_id, args.hf_root, manifest_entries
    )
    # Upload everything under hf_root/
    upload_to_hf(
        repo_id=args.repo_id,
        local_dir=out_dir,
        path_in_repo=args.hf_root,
        token=token,
        private=args.private,
    )
    print("Done")
    print(f"Local artifacts: {out_dir}")
    print(
        f"Uploaded to: https://huggingface.co/{args.repo_id}/tree/main/{args.hf_root}"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {sha_file}")


if __name__ == "__main__":
    main()
