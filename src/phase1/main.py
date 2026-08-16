import torch

from src.phase1.device import get_device
from src.phase1.vae import load_vae
from src.phase1.vae import freeze_vae
from src.phase1.vae import encode_image
from src.phase1.vae import decode_latent
from src.phase1.image_utils import load_image
from src.phase1.output_utils import save_image
from src.phase1.output_utils import reconstruction_error

from src.phase2.attack import run_attack

from config import (
    TEST_IMAGE,
    RECONSTRUCTION_IMAGE,
    PROTECTED_IMAGE,
)


def main():

    print("=" * 60)
    print("NoiseGuard")
    print("=" * 60)

    print(f"PyTorch : {torch.__version__}")

    device = get_device()

    vae = load_vae(device)
    freeze_vae(vae)

    total_parameters = sum(
        parameter.numel()
        for parameter in vae.parameters()
    )

    print(f"\nParameters : {total_parameters:,}")

    print("\n✓ Ready for inference")

    # Load image

    image = load_image(
        TEST_IMAGE,
        device
    )

    print("\n✓ Image Loaded")

    print(f"Shape : {image.shape}")
    print(f"Min   : {image.min().item():.3f}")
    print(f"Max   : {image.max().item():.3f}")

    # Encode

    latent = encode_image(
        vae,
        image
    )

    print("\n✓ Latent Created")

    print(f"Shape : {latent.shape}")
    print(f"Mean  : {latent.mean().item():.4f}")
    print(f"Std   : {latent.std().item():.4f}")

    # Decode

    reconstructed = decode_latent(
        vae,
        latent
    )

    print("\n✓ Reconstruction Complete")

    # Save reconstruction

    save_image(
        reconstructed,
        RECONSTRUCTION_IMAGE
    )

    # Reconstruction error

    mse = reconstruction_error(
        image,
        reconstructed
    )

    print(f"\nReconstruction MSE : {mse:.6f}")

    # -----------------------------------------------------
    # Phase 2 - PGD Attack
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("Phase 2 - Encoder PGD Attack")
    print("=" * 60)

    (
        protected_image,
        loss_history,
        original_latent,
        protected_latent,
        final_distance,
    ) = run_attack(
        vae,
        image,
        epsilon=8 / 255,
        alpha=2 / 255,
        steps=50,
    )

    print("\n✓ PGD Attack Complete")

    print(
        f"Initial latent distance : "
        f"{loss_history[0]:.6f}"
    )

    print(
        f"Final latent distance   : "
        f"{final_distance:.6f}"
    )

    # Save protected image

    save_image(
        protected_image,
        PROTECTED_IMAGE
    )

    print(
        f"\n✓ Protected image saved:"
        f"\n{PROTECTED_IMAGE}"
    )

    print("\n✓ Phase 2 Complete")


if __name__ == "__main__":
    main()