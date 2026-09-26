import math
import torch
from PIL import Image


def tensor_to_pil(tensor):
    x = tensor.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1) / 2).squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((x * 255).round().astype('uint8'), 'RGB')


def pil_to_tensor(image, device, dtype=torch.float16):
    image = image.convert('RGB').resize((256, 256))
    values = torch.from_numpy(__import__('numpy').array(image)).float() / 255.0
    values = values.permute(2, 0, 1).unsqueeze(0)
    values = values * 2.0 - 1.0
    return values.to(device=device, dtype=dtype)


def evaluate_downstream_reconstruction(original_image, protected_image, generated_pil, device):
    """Compare the open-source diffusion reconstruction with the original/protected input."""
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    original = original_image.float()
    protected = protected_image.float()
    generated = pil_to_tensor(generated_pil, device, dtype=dtype).float()

    with torch.no_grad():
        mse_original = torch.mean((original - generated) ** 2).item()
        mse_protected = torch.mean((protected - generated) ** 2).item()
        mse_input_change = torch.mean((original - protected) ** 2).item()

        psnr_original = 10.0 * math.log10(4.0 / max(mse_original, 1e-12))
        psnr_protected = 10.0 * math.log10(4.0 / max(mse_protected, 1e-12))

    return {
        'model': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
        'method': 'Stable Diffusion Img2Img',
        'generatedVsOriginalMse': round(mse_original, 8),
        'generatedVsOriginalPsnrDb': round(psnr_original, 4),
        'generatedVsProtectedMse': round(mse_protected, 8),
        'generatedVsProtectedPsnrDb': round(psnr_protected, 4),
        'originalVsProtectedMse': round(mse_input_change, 8),
    }
