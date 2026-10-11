import argparse
from pathlib import Path

import torch

from inference import sample_with_diagnostics, save_generated_images
from model import UNet
from reverse_step import make_diffusion_schedule
from train import train_model


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested_device)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and sample a FashionMNIST DDPM."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train the diffusion model.")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--data-dir", default="./data")
    train_parser.add_argument("--output-dir", default="./outputs")
    train_parser.add_argument(
        "--checkpoint",
        default="./checkpoints/fashion_mnist_ddpm.pt",
    )
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    sample_parser = commands.add_parser(
        "sample",
        help="Generate images from a checkpoint.",
    )
    sample_parser.add_argument(
        "--checkpoint",
        default="./checkpoints/fashion_mnist_ddpm.pt",
    )
    sample_parser.add_argument("--num-samples", type=int, default=64)
    sample_parser.add_argument("--output-dir", default="./outputs")
    sample_parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    device = resolve_device(args.device)

    if args.command == "train":
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1.")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1.")
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            device=device,
            num_workers=args.num_workers,
        )
        return

    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1.")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = UNet(
        in_ch=checkpoint["in_channels"],
        channels=tuple(checkpoint["channels"]),
        time_dim=checkpoint["time_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    schedule = make_diffusion_schedule(
        num_timesteps=checkpoint["num_timesteps"],
        device=device,
    )
    images, intermediates, _ = sample_with_diagnostics(
        model=model,
        schedule=schedule,
        n_samples=args.num_samples,
        device=device,
    )
    save_generated_images(images, intermediates, args.output_dir)
    print(f"Saved generated images and denoising trajectory to {args.output_dir}")


if __name__ == "__main__":
    main()
