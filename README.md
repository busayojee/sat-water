# SEGMENTATION OF WATER BODIES FROM SATELLITE IMAGES (sat-water)

## Introduction
Satellite imagery is a rich source of information, and the accurate segmentation of water bodies is crucial for understanding environmental patterns and changes over time. This project aims to provide a reliable and efficient tool for extracting water regions from raw satellite images.

This repository supports two workflows:
1. **Library usage**: install with `pip` and run inference (pretrained weights downloaded on-demand).
2. **Training workflow**: train your own models using the included preprocessing + training pipeline.

---

## Dataset

The dataset for this project is gotten here [kaggle.com](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies). It consists of jpeg images of water bodies taken by satellites and their mask. More details of the dataset is provided on the website.

---

## Installation

### As a library
```bash
pip install sat-water
```

To run inference/training you must install the TensorFlow extras:

```bash
pip install "sat-water[tf]"
```

### From source (development)
```bash
git clone https://github.com/busayojee/sat-water.git
cd sat-water
pip install -e .
```

> Note: `sat-water` sets `TF_USE_LEGACY_KERAS=1` and `SM_FRAMEWORK=tf.keras` by default at import time to keep `segmentation-models` compatible.

---

## Pretrained models

Pretrained weights are hosted on Hugging Face and downloaded at inference time with SHA256 integrity verification.

Default weights repo:
- `busayojee/sat-water-weights`

Override weights source:
```bash
export SATWATER_WEIGHTS_REPO="busayojee/sat-water-weights"
export SATWATER_WEIGHTS_REV="main"
```

### Available model keys

This project was trained on 2 models. The <b>UNET</b> with no backbone and the UNET with a <b>RESNET34</b> backbone of which 2 different models were trained on different sizes of images and also different hyperparameters. 

| Model key | Architecture | Input size | Notes |
|---|---|---:|---|
| `resnet34_256` | UNet + ResNet34 backbone | 256×256 | Best speed/quality tradeoff |
| `resnet34_512` | UNet + ResNet34 backbone | 512×512 | Higher-res boundaries; slower |
| `unet` | UNet (no backbone) | 128×128 | Currently unavailable in weights repo |

---

## Quickstart (library inference)

```python
from satwater.inference import segment_image

res = segment_image(
    "path/to/image.jpg",
    model="resnet34_512",      # or "resnet34_256"
    return_overlay=True,
    show=False,
)

mask = res.masks["resnet34_512"]         # (H, W, 1)
overlay = res.overlays["resnet34_512"]   # (H, W, 3)
```

---

## Inference API

`segment_image(...)` is the recommended entrypoint for package users.

### Parameters (commonly used)

- `image_path` *(str)*: path to an input image (`.jpg`, `.png`, etc.)
- `model` *(str)*: one of `resnet34_256`, `resnet34_512` (and `unet` once available)
- `return_overlay` *(bool)*: whether to return an overlay image (original image + blended water mask)
- `show` *(bool)*: whether to display the result via matplotlib (useful in notebooks / local runs)

### Weights source / versioning

- `repo_id` *(str, optional)*: Hugging Face repo containing weights (defaults to `SATWATER_WEIGHTS_REPO`)
- `revision` *(str, optional)*: branch / tag / commit (defaults to `SATWATER_WEIGHTS_REV`)
- `save_dir` *(str | Path | None, optional)*: output directory (if supported in your local version).  
  If you want saving, you can always do it manually from the returned arrays (example below).

#### Manual saving 

```python
from PIL import Image
import numpy as np

Image.fromarray((mask.squeeze(-1) * 255).astype(np.uint8)).save("mask.png")
Image.fromarray(overlay).save("overlay.png")
```

---

## Training history (reference)

The plots below are from historical runs in this repository and are provided to show convergence behavior.

| UNet (baseline) | ResNet34-UNet (256×256) | ResNet34-UNet (512×512) |
|:--:|:--:|:--:|
| <img width="260" alt="UNet History" src="https://github.com/busayojee/sat-water/blob/main/assets/results/history_unet.png"> | <img width="260" alt="ResNet34 256 History" src="https://github.com/busayojee/sat-water/blob/main/assets/results/history_resnet34.png"> | <img width="260" alt="ResNet34 512 History" src="https://github.com/busayojee/sat-water/blob/main/assets/results/historyresnet34(2).png"> |

---

## Inference examples

Qualitative predictions produced by the three models.

| UNet | ResNet34-UNet (256×256) | ResNet34-UNet (512×512) |
|:--:|:--:|:--:|
| <img width="260" alt="UNet Prediction" src="https://github.com/busayojee/sat-water/blob/main/assets/results/prediciton_unet.png"> | <img width="260" alt="ResNet34 256 Prediction" src="https://github.com/busayojee/sat-water/blob/main/assets/results/prediciton_resnet34.png"> | <img width="260" alt="ResNet34 512 Prediction" src="https://github.com/busayojee/sat-water/blob/main/assets/results/prediciton_resnet34(2).png"> |

---

## Single test instance (end-to-end)

Using all models to predict a single test instance.

| Test Image | Prediction |
|:--:|:--:|
| <img width="300" alt="Test Image" src="https://github.com/busayojee/sat-water/blob/main/assets/results/test2.jpg"> | <img width="300" alt="Prediction" src="https://github.com/busayojee/sat-water/blob/main/assets/results/prediciton_test.png"> |

Label overlay of the best prediction (ResNet34-UNet 512×512 in that run):

<img width="320" alt="Overlay" src="https://github.com/busayojee/sat-water/blob/main/assets/results/test2.png">

---

## Train your own model

### Preprocessing

```python
from satwater.preprocess import Preprocess

train_ds, val_ds, test_ds = Preprocess.data_load(
    dataset_dir="path/to/dataset",
    masks_dir="/Masks",
    images_dir="/Images",
    split=(0.7, 0.2, 0.1),
    shape=(256, 256),
    batch_size=16,
    channels=3,
)
```

### Training (UNet baseline)
```python
from satwater.models import Unet

history = Unet.train(
    train_ds,
    val_ds,
    shape=(128, 128, 3),
    n_classes=2,
    lr=1e-4,
    loss=Unet.loss,
    metrics=Unet.metrics,
    name="unet",
)
```

### Training (ResNet34-UNet)
```python
from satwater.models import BackboneModels

bm = BackboneModels("resnet34", train_ds, val_ds, test_ds, name="resnet34_256")
bm.build_model(n_classes=2, n=1, lr=1e-4)
history = bm.train()
```

> For a 512×512 run, load a second dataset with `shape=(512, 512)` and use a different model name (e.g. `resnet34_512`) to keep artifacts separate.


### Inference
To run inference for UNET

```
inference_u = Inference(model="path/to/model",name="unet")
inference_u.predict_ds(test_ds)
```

for RESNET 1 and 2

```
inference_r = Inference(model="path/to/model",name="resnet34")
inference_r.predict_ds(test_ds)

inference_r2 = Inference(model="path/to/model",name="resnet34(2)")
inference_r2.predict_ds(test_ds1)
```

For all 3 models together

```
models={"unet":"path/to/model1", "resnet34":"path/to/model2", "resnet34(2)":"path/to/model3"}
inference_multiple = Inference(model=models)
inference_multiple.predict_ds(test_ds)
```

## CLI (optional)

If you included the `scripts/` folder in your package/repo, you can run the scripts directly.

### Training CLI

UNet:

```bash
python scripts/train.py   --dataset path/to/dataset   --image-folder /Images   --mask-folder /Masks   --shape 128,128,3   --batch-size 16   --split 0.2,0.1   --channels 3   --model unet   --name unet   --epochs 100   --lr 1e-4
```

ResNet34-UNet (256):

```bash
python scripts/train.py   --dataset path/to/dataset   --image-folder /Images   --mask-folder /Masks   --shape 256,256,3   --batch-size 8   --split 0.2,0.1   --channels 3   --model resnet34   --name resnet34_256   --epochs 100   --lr 1e-4
```

ResNet34-UNet (512):

```bash
python scripts/train.py   --dataset path/to/dataset   --image-folder /Images   --mask-folder /Masks   --shape 512,512,3   --batch-size 4   --split 0.2,0.1   --channels 3   --model resnet34(2)   --name resnet34_512   --epochs 100   --lr 1e-4
```

### Inference CLI

Single model:

```bash
python scripts/infer.py   --image path/to/image.jpg   --model path/to/model.keras   --name unet   --out prediction
```

Multiple models:

```bash
python scripts/infer.py   --image path/to/image.jpg   --models "unet=path/to/unet.keras,resnet34=path/to/resnet34.keras,resnet34(2)=path/to/resnet34_2.keras"   --out prediction
```

### Upload weights to Hugging Face (optional)

```bash
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

python scripts/weights.py   --repo-id user/repo   --hf-root weights   --out-dir dist/weights   --model unet=path/to/unet.keras@128,128,3   --model resnet34_256=path/to/resnet34_256.keras@256,256,3   --model resnet34_512=path/to/resnet34_512.keras@512,512,3
```

---

## Contributing

Contributions are welcome — especially around:
- adding/refreshing pretrained weights (including UNet)
- improving inference UX (CLI, batch inference, better overlays)
- expanding tests and CI matrix
- model evaluation and benchmarking on additional datasets

### How to contribute
1. Fork the repo
2. Create a feature branch:
   ```bash
   git checkout -b feat/my-change
   ```
3. Run checks locally:
   ```bash
   pytest -q
   ruff check .
   ruff format .
   ```
4. Open a pull request with a short summary + screenshots (if changing inference output)

If you’re reporting a bug, please include:
- OS + Python version
- TensorFlow version
- full traceback + a minimal repro snippet

---


