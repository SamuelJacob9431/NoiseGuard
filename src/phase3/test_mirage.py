import torch
from pathlib import Path

from src.phase1.device import get_device
from src.phase1.image_utils import load_image
from src.phase1.output_utils import save_image

from src.phase3.mirage import mirage_attack


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / "images"
    / "test_image.jpg"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "mirage_protected.jpg"
)


def main():

    print("=" * 60)
    print("NoiseGuard - MIRAGE Test")
    print("=" * 60)

    device = get_device()

    image = load_image(
        IMAGE_PATH,
        device
    )

    print(
        f"\nImage shape : {image.shape}"
    )

    (
        mirage_image,
        loss_history,
        initial_score,
        final_score,
    ) = mirage_attack(
        image,
        device,
        epsilon=8 / 255,
        alpha=1 / 255,
        steps=30,
    )

    save_image(
        mirage_image,
        OUTPUT_PATH
    )

    print(
        f"\nInitial similarity : "
        f"{initial_score:.6f}"
    )

    print(
        f"Final similarity   : "
        f"{final_score:.6f}"
    )

    print(
        f"Change             : "
        f"{final_score - initial_score:.6f}"
    )

    print(
        "\n✓ MIRAGE image saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()