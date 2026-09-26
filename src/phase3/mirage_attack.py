import torch

from src.phase3.mirage_loss import mirage_loss


def mirage_attack(
    model,
    original_image,
    target_embedding,
    epsilon=8 / 255,
    alpha=2 / 255,
    steps=20
):
    model.eval()

    original_image = original_image.detach()

    adversarial_image = (
        original_image.clone().detach()
    )

    loss_history = []

    for step in range(steps):

        adversarial_image.requires_grad_(True)

        image_embedding = model.encode_image(
            adversarial_image
        )

        loss = mirage_loss(
            image_embedding,
            target_embedding
        )

        loss_history.append(
            loss.item()
        )

        model.zero_grad(set_to_none=True)

        if adversarial_image.grad is not None:
            adversarial_image.grad.zero_()

        loss.backward()

        with torch.no_grad():

            gradient = adversarial_image.grad.sign()

            adversarial_image = (
                adversarial_image
                - alpha * gradient
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

        adversarial_image = (
            adversarial_image.detach()
        )

    return (
        adversarial_image,
        loss_history
    )