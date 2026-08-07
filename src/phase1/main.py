import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))
from device import get_device
from vae import load_vae
from vae import freeze_vae
from vae import encode_image
from vae import decode_latent

from image_utils import load_image

from output_utils import save_image
from output_utils import reconstruction_error

from config import (
    TEST_IMAGE,
    RECONSTRUCTION_IMAGE,
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

    # -----------------------------------------------------
    # Load Image
    # -----------------------------------------------------

    image = load_image(
        TEST_IMAGE,
        device
    )

    print("\n✓ Image Loaded")

    print(f"Shape : {image.shape}")
    print(f"Min   : {image.min().item():.3f}")
    print(f"Max   : {image.max().item():.3f}")

    # -----------------------------------------------------
    # Encode
    # -----------------------------------------------------

    latent = encode_image(
        vae,
        image
    )

    print("\n✓ Latent Created")

    print(f"Shape : {latent.shape}")
    print(f"Mean  : {latent.mean().item():.4f}")
    print(f"Std   : {latent.std().item():.4f}")

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    reconstructed = decode_latent(
        vae,
        latent
    )

    print("\n✓ Reconstruction Complete")

    # -----------------------------------------------------
    # Save Output
    # -----------------------------------------------------

    save_image(
        reconstructed,
        RECONSTRUCTION_IMAGE
    )

    # -----------------------------------------------------
    # Reconstruction Error
    # -----------------------------------------------------

    mse = reconstruction_error(
        image,
        reconstructed
    )

    print(f"\nReconstruction MSE : {mse:.6f}")

    print("\n✓ Phase 1 Complete")


if __name__ == "__main__":
    main()