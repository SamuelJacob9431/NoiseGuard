from pathlib import Path

import numpy as np
from PIL import Image
import torch


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a tensor in [-1, 1] to a PIL image.
    Expected shape: [1, C, H, W]
    """

    tensor = tensor.detach()

    tensor = (tensor / 2) + 0.5

    tensor = tensor.clamp(0, 1)

    tensor = tensor.squeeze(0)

    tensor = tensor.permute(1, 2, 0)

    tensor = tensor.cpu().float().numpy()

    tensor = (tensor * 255).astype(np.uint8)

    return Image.fromarray(tensor)


def save_image(tensor: torch.Tensor, output_path):
    """
    Saves a tensor as an image.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image = tensor_to_pil(tensor)

    image.save(output_path)

    print(f"\n✓ Image saved to:\n{output_path}")


def reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
):
    """
    Computes Mean Squared Error.
    """

    mse = torch.mean(
        (original.float() - reconstructed.float()) ** 2
    )

    return mse.item()