from diffusers import AutoencoderKL
import torch


MODEL_ID = "runwayml/stable-diffusion-v1-5"

def load_vae(device):

    print("\nLoading Stable Diffusion VAE...")

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    vae = AutoencoderKL.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=dtype
    )

    vae = vae.to(device)

    return vae


def freeze_vae(vae):

    vae.eval()

    for parameter in vae.parameters():
        parameter.requires_grad = False

    print("✓ VAE Frozen")

    return vae

def encode_image(vae, image):

    with torch.no_grad():

        latent_distribution = vae.encode(image).latent_dist

        latent = latent_distribution.sample()

    return latent