import torch


def latent_distance(
    original_latent: torch.Tensor,
    adversarial_latent: torch.Tensor,
) -> torch.Tensor:

    return torch.mean(
        (original_latent - adversarial_latent) ** 2
    )