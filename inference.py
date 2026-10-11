from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from reverse_step import DiffusionSchedule, posterior_mean


@torch.no_grad()
def sample_with_diagnostics(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    n_samples: int = 64,
    device: torch.device | str = "cpu",
    save_steps: list[int] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], list[float]]:
    """Generate images and retain selected denoising states and norms."""
    device = torch.device(device)
    if save_steps is None:
        last_timestep = schedule.num_timesteps - 1
        save_steps = sorted(
            {
                last_timestep,
                round(last_timestep * 0.75),
                round(last_timestep * 0.5),
                round(last_timestep * 0.25),
                0,
            }
        )

    was_training = model.training
    model.eval()
    images = torch.randn(n_samples, 1, 28, 28, device=device)
    intermediates = {}
    norms = []

    for timestep in reversed(range(schedule.num_timesteps)):
        timesteps = torch.full(
            (n_samples,),
            timestep,
            device=device,
            dtype=torch.long,
        )
        predicted_noise = model(images, timesteps)
        mean = posterior_mean(images, timestep, predicted_noise, schedule)
        if timestep > 0:
            images = mean + schedule.beta_tilde[timestep].sqrt() * torch.randn_like(
                images
            )
        else:
            images = mean

        if timestep in save_steps:
            intermediates[timestep] = images.clone()
        norms.append(images.norm().item())

    model.train(was_training)
    return images, intermediates, norms


def save_generated_images(
    images: torch.Tensor,
    intermediates: dict[int, torch.Tensor],
    output_dir: str = "./outputs",
) -> None:
    """Save a generated image grid and a grid of selected denoising states."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    display_images = (images.clamp(-1, 1) + 1) / 2
    grid = make_grid(display_images.cpu(), nrow=8, padding=2)
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(
        grid.permute(1, 2, 0).squeeze(-1),
        cmap="gray",
    )
    axis.axis("off")
    axis.set_title("Generated FashionMNIST images")
    figure.savefig(
        output_path / "generated_samples.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    ordered_states = sorted(intermediates.items(), reverse=True)
    figure, axes = plt.subplots(
        1,
        len(ordered_states),
        figsize=(3 * len(ordered_states), 3),
        squeeze=False,
    )
    for axis, (timestep, state) in zip(axes[0], ordered_states):
        state_grid = make_grid(
            (state[:16].clamp(-1, 1) + 1) / 2,
            nrow=4,
            padding=1,
        )
        axis.imshow(state_grid.permute(1, 2, 0).squeeze(-1), cmap="gray")
        axis.set_title(f"t = {timestep}")
        axis.axis("off")
    figure.suptitle("Denoising trajectory")
    figure.tight_layout()
    figure.savefig(
        output_path / "denoising_trajectory.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)
