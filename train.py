from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataset import get_dataloader
from model import UNet
from reverse_step import make_diffusion_schedule, q_sample


def train_model(
    epochs: int = 50,
    batch_size: int = 128,
    data_dir: str = "./data",
    output_dir: str = "./outputs",
    checkpoint_path: str = "./checkpoints/fashion_mnist_ddpm.pt",
    device: torch.device | str = "cpu",
    num_workers: int = 0,
) -> Path:
    """Train a noise-predicting U-Net and save its weights and loss plot."""
    device = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    schedule = make_diffusion_schedule(device=device)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    losses = []
    gradient_norms = []

    print(f"Training on {device}")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, _ in progress:
            images = images.to(device)
            timesteps = torch.randint(
                0,
                schedule.num_timesteps,
                (images.shape[0],),
                device=device,
            )
            noise = torch.randn_like(images)
            noisy_images = q_sample(images, timesteps, noise, schedule)
            predicted_noise = model(noisy_images, timesteps)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            gradient_norms.append(total_norm.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_loss = epoch_loss / num_batches
        losses.append(average_loss)
        print(f"Epoch {epoch + 1} - average loss: {average_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": 1,
            "channels": (32, 64, 128),
            "time_dim": 128,
            "num_timesteps": schedule.num_timesteps,
        },
        checkpoint,
    )

    figure, (loss_axis, norm_axis) = plt.subplots(1, 2, figsize=(14, 5))
    loss_axis.plot(losses)
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Training loss")
    norm_axis.plot(gradient_norms, alpha=0.3)
    norm_axis.set_xlabel("Training step")
    norm_axis.set_ylabel("Gradient norm")
    norm_axis.set_title("Gradient norms")
    figure.tight_layout()
    figure.savefig(output_path / "training_curves.png", dpi=150)
    plt.close(figure)

    print(f"Saved model checkpoint to {checkpoint}")
    print(f"Saved training curves to {output_path / 'training_curves.png'}")
    return checkpoint
