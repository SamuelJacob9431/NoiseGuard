from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms


IMAGE_SIZE = 256


def load_image(image_path, device):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    dtype = (
    torch.float16
    if device.type == "cuda"
    else torch.float32
)

    image_tensor = image_tensor.to(
    device=device,
    dtype=dtype
     )

    return image_tensor