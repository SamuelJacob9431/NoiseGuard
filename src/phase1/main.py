import torch

from device import get_device
from vae import load_vae


def main():

    print("=" * 60)
    print("NoiseGuard")
    print("=" * 60)

    print(torch.__version__)

    device = get_device()

    vae = load_vae(device)

    print()

    print("Ready.")

if __name__ == "__main__":
    main()