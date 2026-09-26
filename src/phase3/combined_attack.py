import torch

from src.phase1.vae import encode_image, decode_latent
from src.phase2.pgd import pgd_attack


def pgd_then_mirage(
    vae,
    mirage_model,
    image,
    pgd_epsilon=8 / 255,
    pgd_alpha=2 / 255,
    pgd_steps=50,
    mirage_epsilon=8 / 255,
    mirage_alpha=1 / 255,
    mirage_steps=30,
):
    """
    NoiseGuard pipeline:
        original image -> latent PGD -> MIRAGE representation optimization

    Returns:
        final_image, pgd_image, protected_reconstruction,
        pgd_loss_history, mirage_loss_history,
        original_latent, protected_latent,
        final_latent_distance, reconstruction_mse
    """

    vae.eval()
    original_image = image.detach().clone()

    # Stage 1: latent-space PGD
    with torch.no_grad():
        original_latent = encode_image(vae, original_image)

    pgd_image, pgd_loss_history = pgd_attack(
        vae,
        original_image,
        original_latent,
        epsilon=pgd_epsilon,
        alpha=pgd_alpha,
        steps=pgd_steps,
    )

    # Stage 2: MIRAGE representation optimization
    mirage_model.model.eval()
    adversarial = pgd_image.detach().clone()
    mirage_loss_history = []

    for step in range(mirage_steps):
        adversarial = adversarial.detach().requires_grad_(True)

        score = mirage_model.target_similarity(adversarial)
        mirage_loss_history.append(score.item())

        mirage_model.model.zero_grad(set_to_none=True)

        if adversarial.grad is not None:
            adversarial.grad.zero_()

        score.backward()

        with torch.no_grad():
            gradient = adversarial.grad.sign()

            adversarial = adversarial + mirage_alpha * gradient

            # Keep the final image within the L-infinity budget
            # relative to the original image.
            perturbation = adversarial - original_image
            perturbation = torch.clamp(
                perturbation,
                -mirage_epsilon,
                mirage_epsilon,
            )

            adversarial = original_image + perturbation
            adversarial = torch.clamp(adversarial, -1.0, 1.0)

    final_image = adversarial.detach()

    # Verification through the frozen VAE
    with torch.no_grad():
        protected_latent = encode_image(vae, final_image)
        protected_reconstruction = decode_latent(vae, protected_latent)

        final_latent_distance = torch.mean(
            (original_latent - protected_latent) ** 2
        ).item()

        reconstruction_mse = torch.mean(
            (original_image.float() - protected_reconstruction.float()) ** 2
        ).item()

    return (
        final_image,
        pgd_image,
        protected_reconstruction,
        pgd_loss_history,
        mirage_loss_history,
        original_latent,
        protected_latent,
        final_latent_distance,
        reconstruction_mse,
    )
