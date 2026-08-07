import torch

from device import get_device
from vae import load_vae
from vae import freeze_vae


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

if __name__ == "__main__":
    main()