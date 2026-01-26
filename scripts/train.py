import argparse

from satwater.models import BackboneModels, Unet
from satwater.preprocess import Preprocess
from satwater.utils import parse_shape, set_seed


def build_parser():
    p = argparse.ArgumentParser(description="Train sat-water segmentation models.")
    p.add_argument("--dataset", required=True, help="Path to dataset root folder")
    p.add_argument(
        "--image-folder", default="/Images", help="Image folder name (e.g. /Images)"
    )
    p.add_argument(
        "--mask-folder", default="/Masks", help="Mask folder name (e.g. /Masks)"
    )

    p.add_argument(
        "--shape", default="128,128,3", help="Input shape as H,W,C (default: 128,128,3)"
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--split", default="0.2,0.1", help="val,test split ratios (default: 0.2,0.1)"
    )
    p.add_argument("--channels", type=int, default=3)

    p.add_argument(
        "--model", default="unet", choices=["unet", "resnet34", "resnet34(2)"]
    )
    p.add_argument("--name", default="model", help="Output model name prefix")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)

    return p


def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    shape = parse_shape(args.shape)
    split_vals = [float(x) for x in args.split.split(",")]
    if len(split_vals) != 2:
        raise ValueError("--split must look like '0.2,0.1' (val,test)")

    train_ds, val_ds, test_ds = Preprocess.data_load(
        dataset=args.dataset,
        mask_folder=args.mask_folder,
        image_folder=args.image_folder,
        split=split_vals,
        shape=shape,
        batch_size=args.batch_size,
        channels=args.channels,
    )

    if args.model == "unet":
        # uses custom unet
        history = Unet.train(
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=args.epochs,
            shape=shape,
            n_classes=2,
            lr=args.lr,
            name=args.name,
        )
        try:
            Unet.plot_history(history, epochs=args.epochs, model=args.name)
        except Exception:
            pass
    else:
        # Uses segmentation_models backbone UNet
        bm = BackboneModels(args.model, train_ds, val_ds, test_ds, name=args.name)
        bm.build_model(n_classes=2, n=1, lr=args.lr)
        bm.train()


if __name__ == "__main__":
    main()
