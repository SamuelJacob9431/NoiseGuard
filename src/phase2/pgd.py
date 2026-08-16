import torch

from src.phase1.vae import encode_image
from src.phase2.losses import latent_distance


def pgd_attack(
    vae,
    original_image,
    original_latent,
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=50,
):
    vae.eval()

    adversarial_image = original_image.detach().clone()
    loss_history = []

    for step in range(steps):

        adversarial_image = (
            adversarial_image
            .detach()
            .requires_grad_(True)
        )

        adversarial_latent = encode_image(
            vae,
            adversarial_image,
            requires_grad=True
        )

        loss = latent_distance(
            original_latent,
            adversarial_latent
        )

        loss_history.append(loss.item())

        vae.zero_grad(set_to_none=True)

        loss.backward()

        with torch.no_grad():

            gradient = adversarial_image.grad.sign()

            adversarial_image = (
                adversarial_image
                + alpha * gradient
            )

            perturbation = (
                adversarial_image
                - original_image
            )

            perturbation = torch.clamp(
                perturbation,
                -epsilon,
                epsilon
            )

            adversarial_image = (
                original_image
                + perturbation
            )

            adversarial_image = torch.clamp(
                adversarial_image,
                -1.0,
                1.0
            )

    return (
        adversarial_image.detach(),
        loss_history,
    )