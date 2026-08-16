import torch

from .losses import latent_distance


def pgd_attack(
    vae,
    original_image,
    original_latent,
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=50,
):

    adversarial_image = (
        original_image
        .detach()
        .clone()
    )

    loss_history = []

    for step in range(steps):

        adversarial_image.requires_grad_(True)

        adversarial_latent = vae.encode(
            adversarial_image
        ).latent_dist.mean

        loss = latent_distance(
            original_latent,
            adversarial_latent
        )

        loss_history.append(
            loss.item()
        )

        vae.zero_grad(set_to_none=True)

        if adversarial_image.grad is not None:
            adversarial_image.grad.zero_()

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
        loss_history
    )