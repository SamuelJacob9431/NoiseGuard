import torch
from pathlib import Path
from device import get_device
from vae import load_vae
from vae import encode_image
from vae import freeze_vae
from image_utils import load_image


def main():

    print("=" * 60)
    print("NoiseGuard")
    print("=" * 60)

    print(torch.__version__)

    device = get_device()

    vae = load_vae(device)
    vae = freeze_vae(vae)
    
    total_parameters = sum(
        parameter.numel()
        for parameter in vae.parameters()
    )

    

    print(f"Parameters : {total_parameters:,}")

    print("\n✓ Ready for inference")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    IMAGE_PATH = PROJECT_ROOT / "images" / "test_image.jpg"

    image = load_image(IMAGE_PATH, device)
    print("\nImage Loaded Successfully")
    print(f"Shape : {image.shape}")
    print(f"Min   : {image.min().item():.3f}")
    print(f"Max   : {image.max().item():.3f}")

    latent = encode_image(vae,image)
    print("\nLatent Created")

    print(f"Shape : {latent.shape}")

    print(f"Mean : {latent.mean().item():.4f}")

    print(f"Std : {latent.std().item():.4f}")
if __name__ == "__main__":
    main()