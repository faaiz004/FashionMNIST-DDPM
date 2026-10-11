
from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    """Precomputed noise schedule tensors, stored on the target device."""

    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor
    alpha_bar_prev: torch.Tensor
    beta_tilde: torch.Tensor

    @property
    def num_timesteps(self) -> int:
        return self.beta.shape[0]


def make_diffusion_schedule(
    num_timesteps: int = 1000,
    device: torch.device | str = "cpu",
) -> DiffusionSchedule:
    """Build the linear variance schedule used by the forward process."""
    beta = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]])
    beta_tilde = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * beta
    return DiffusionSchedule(
        beta=beta,
        alpha=alpha,
        alpha_bar=alpha_bar,
        alpha_bar_prev=alpha_bar_prev,
        beta_tilde=beta_tilde,
    )


def q_sample(
    x_0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Add noise to clean images at selected diffusion timesteps."""
    signal_weight = schedule.alpha_bar[timesteps].sqrt().reshape(-1, 1, 1, 1)
    noise_weight = (1.0 - schedule.alpha_bar[timesteps]).sqrt().reshape(
        -1, 1, 1, 1
    )
    return signal_weight * x_0 + noise_weight * noise


def predict_x0(
    x_t: torch.Tensor,
    timestep: int,
    predicted_noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Estimate the clean image from a noisy image and predicted noise."""
    alpha_bar_t = schedule.alpha_bar[timestep]
    return (
        x_t - (1.0 - alpha_bar_t).sqrt() * predicted_noise
    ) / alpha_bar_t.sqrt()


def posterior_mean(
    x_t: torch.Tensor,
    timestep: int,
    predicted_noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Compute the DDPM reverse-process posterior mean."""
    alpha_t = schedule.alpha[timestep]
    alpha_bar_t = schedule.alpha_bar[timestep]
    beta_t = schedule.beta[timestep]
    return (
        x_t - beta_t / (1.0 - alpha_bar_t).sqrt() * predicted_noise
    ) / alpha_t.sqrt()


def reverse_step(
    x_t: torch.Tensor,
    timestep: int,
    predicted_noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Perform one ancestral sampling step from x_t to x_(t-1)."""
    mean = posterior_mean(x_t, timestep, predicted_noise, schedule)
    if timestep == 0:
        return mean
    return mean + schedule.beta_tilde[timestep].sqrt() * torch.randn_like(x_t)


def q_posterior_mean(
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    timestep: int,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Compute the true posterior mean when x_0 is known."""
    alpha_bar_t = schedule.alpha_bar[timestep]
    alpha_bar_prev_t = schedule.alpha_bar_prev[timestep]
    beta_t = schedule.beta[timestep]
    alpha_t = schedule.alpha[timestep]
    weight_x0 = alpha_bar_prev_t.sqrt() * beta_t / (1.0 - alpha_bar_t)
    weight_xt = (
        alpha_t.sqrt() * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
    )
    return weight_x0 * x_0 + weight_xt * x_t
