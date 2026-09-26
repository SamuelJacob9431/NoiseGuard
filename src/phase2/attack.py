import torch

from src.phase1.vae import encode_image, decode_latent
from src.phase2.pgd import pgd_attack


def run_attack(
    vae,
    image,
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=50,
):
    vae.eval()

    with torch.no_grad():
        original_latent = encode_image(
            vae,
            image
        )

    protected_image, loss_history = pgd_attack(
        vae,
        image,
        original_latent,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
    )

    with torch.no_grad():

        protected_latent = encode_image(
            vae,
            protected_image
        )

        protected_reconstruction = decode_latent(
            vae,
            protected_latent
        )

    final_distance = torch.mean(
        (original_latent - protected_latent) ** 2
    ).item()

    protected_reconstruction_mse = torch.mean(
        (image - protected_reconstruction) ** 2
    ).item()

    return (
        protected_image,
        protected_reconstruction,
        loss_history,
        original_latent,
        protected_latent,
        final_distance,
        protected_reconstruction_mse,
    )