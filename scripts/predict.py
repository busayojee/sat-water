import argparse

from satwater.inference import Inference
from satwater.utils import parse_models_arg


def build_parser():
    p = argparse.ArgumentParser(description="Run inference on a single image.")
    p.add_argument("--image", required=True, help="Path to an input JPEG image")

    # single model
    p.add_argument("--model", help="Path to a single model file (.keras/.h5)")
    p.add_argument(
        "--name",
        choices=["unet", "resnet34", "resnet34(2)"],
        help="Backbone name for single model",
    )

    # multiple models in one call
    # for example: models "unet=unet.keras,resnet34=r34.keras,resnet34(2)=r34_2.keras"
    p.add_argument("--models", help="Comma-separated key=path pairs")

    p.add_argument("--out", default="prediction", help="Output filename stem")
    return p


def main():
    args = build_parser().parse_args()

    if args.models:
        models: dict[str, str] = parse_models_arg(args.models)
        infer = Inference(model=models)
        infer.predict_inst(args.image, fname=args.out)
        return

    if not args.model or not args.name:
        raise ValueError("Provide either --models OR both --model and --name")

    infer = Inference(model=args.model, name=args.name)
    infer.predict_inst(args.image, fname=args.out)


if __name__ == "__main__":
    main()
