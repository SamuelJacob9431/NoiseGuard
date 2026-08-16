import torch


def latent_distance(original_latent, adversarial_latent):
    #MSE between the original and adversarial latent representations
    return torch.mean(
        (original_latent - adversarial_latent) ** 2
    )