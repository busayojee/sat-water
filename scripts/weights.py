from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from huggingface_hub import HfApi


@dataclass(frozen=True)
class ModelSpec:
    key: str
    path: Path
    input_shape: str | None = None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model_path(p: str) -> Path:
    path = Path(p).expanduser()

    if path.exists():
        return path.resolve()

    # Try common extensions if user omitted them
    for ext in (".keras", ".h5"):
        candidate = Path(str(path) + ext).expanduser()
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Model file not found: {p} (also tried .keras/.h5)")


def export_weights(model_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(model_path), compile=False)

    # Keras standard weights format
    weights_path = out_dir / f"{model_path.stem}.weights.h5"
    model.save_weights(str(weights_path))
    return weights_path


def write_sha_file(out_dir: Path, weights_files: list[Path]) -> Path:
    sha_path = out_dir / "SHA256SUMS.txt"
    lines = []
    for wf in weights_files:
        lines.append(f"{sha256_file(wf)}  {wf.name}")
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha_path


def write_manifest(out_dir: Path, repo_id: str, hf_root: str, entries: dict) -> Path:
    manifest = {
        "repo_id": repo_id,
        "hf_root": hf_root.strip("/"),
        "models": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def upload_to_hf(
    repo_id: str, local_dir: Path, path_in_repo: str, token: str, private: bool
) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_dir),
        path_in_repo=path_in_repo.strip("/"),
        commit_message=f"Upload sat-water weights to {path_in_repo}",
    )


def parse_model_args(model_args: list[str]) -> list[ModelSpec]:
    """
    Accept repeated args like:
      --model unet=~/.../model1
      --model resnet34_256=~/.../model2
      --model resnet34_512=~/.../model3
    Optional shape metadata:
      --model unet=~/.../model1@128,128,3
    """
    specs: list[ModelSpec] = []
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export Keras weights + upload to Hugging Face (multi-model)."
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
    manifest_entries: dict[str, dict] = {}
    weights_files: list[Path] = []

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
