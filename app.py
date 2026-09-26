import base64
import io
import sys
import traceback

import torch
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# Make the cloned repository importable when this file is run from Kaggle.
sys.path.insert(0, "/kaggle/working/noiseguard")

from src.phase1.device import get_device
from src.phase1.vae import load_vae, freeze_vae
from src.phase1.vae import encode_image, decode_latent
from src.phase3.mirage import MirageSurrogate
from src.phase3.combined_attack import pgd_then_mirage
from evaluator import evaluate_protection
from downstream_evaluator import evaluate_downstream_reconstruction, tensor_to_pil
from diffusers import StableDiffusionImg2ImgPipeline


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="NoiseGuard API",
    version="1.0.0",
)

# Demo-only CORS: allow the Netlify frontend to call Kaggle.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Global model state
# ============================================================

device = None
vae = None
mirage_model = None
diffusion_pipe = None


IMAGE_SIZE = 256

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ============================================================
# Model loading
# ============================================================

def load_models():
    global device, vae, mirage_model, diffusion_pipe

    print("=" * 60)
    print("NoiseGuard API - loading models")
    print("=" * 60)

    device = get_device()

    print("\nLoading VAE encoder/decoder...")
    vae = load_vae(device)
    vae = freeze_vae(vae)

    print("\nLoading MIRAGE surrogate...")
    mirage_model = MirageSurrogate(device)

    print("\nLoading open-source Stable Diffusion Img2Img evaluator...")
    diffusion_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
    )
    diffusion_pipe = diffusion_pipe.to(device)
    diffusion_pipe.set_progress_bar_config(disable=True)

    print("\n" + "=" * 60)
    print("✓ NoiseGuard models ready")
    print(f"✓ Device: {device}")
    print("✓ Pipeline: VAE → PGD → MIRAGE → VAE Decoder")
    print("✓ Evaluator: Stable Diffusion v1.5 Img2Img")
    print("=" * 60)


# Load once when FastAPI starts.
load_models()


# ============================================================
# Image helpers
# ============================================================

def upload_to_tensor(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = image_transform(image).unsqueeze(0)

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tensor = tensor.to(
        device=device,
        dtype=dtype,
    )

    return image, tensor


def pil_to_backend_tensor(image):
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor

def tensor_to_data_url(tensor):
    """
    Convert a [-1, 1] BCHW tensor to a JPEG data URL.
    """

    tensor = tensor.detach().float().cpu()

    tensor = tensor.clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0

    tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0)

    array = (tensor.numpy() * 255.0).round().astype("uint8")

    image = Image.fromarray(array, mode="RGB")

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=92,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# Health endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "service": "NoiseGuard API",
        "status": "online",
        "pipeline": "VAE → PGD → MIRAGE → VAE Decoder",
        "device": str(device),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": str(device),
        "vae_loaded": vae is not None,
        "mirage_loaded": mirage_model is not None,
        "diffusion_evaluator_loaded": diffusion_pipe is not None,
    }


# ============================================================
# Main protection endpoint
# ============================================================

@app.post("/protect")
async def protect(file: UploadFile = File(...)):
    if vae is None or mirage_model is None:
        raise HTTPException(
            status_code=503,
            detail="NoiseGuard models are not loaded.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    try:
        print("\n" + "=" * 60)
        print("NEW /protect REQUEST")
        print(f"Filename: {file.filename}")
        print("=" * 60)

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        original_pil, image = upload_to_tensor(image_bytes)

        print(f"✓ Input image: {original_pil.size}")
        print(f"✓ Tensor: {tuple(image.shape)}")

        # ----------------------------------------------------
        # Original VAE encoding / baseline reconstruction
        # ----------------------------------------------------

        with torch.no_grad():
            original_latent = encode_image(
                vae,
                image,
            )

            original_reconstruction = decode_latent(
                vae,
                original_latent,
            )

        latent_mean = original_latent.mean().item()
        latent_std = original_latent.std().item()

        latent_shape = list(original_latent.shape)

        # ----------------------------------------------------
        # PGD → MIRAGE
        # ----------------------------------------------------

        print("\nRunning PGD → MIRAGE...")

        (
            protected_image,
            pgd_image,
            protected_reconstruction,
            pgd_loss_history,
            mirage_loss_history,
            original_latent,
            protected_latent,
            final_latent_distance,
            protected_reconstruction_mse,
        ) = pgd_then_mirage(
            vae=vae,
            mirage_model=mirage_model,
            image=image,
            pgd_epsilon=8 / 255,
            pgd_alpha=2 / 255,
            pgd_steps=50,
            mirage_epsilon=8 / 255,
            mirage_alpha=1 / 255,
            mirage_steps=30,
        )

        # ----------------------------------------------------
        # Backend-only evaluator
        # ----------------------------------------------------

        evaluation = evaluate_protection(
            original_image=image,
            protected_image=protected_image,
            original_latent=original_latent,
            protected_latent=protected_latent,
            original_reconstruction=original_reconstruction,
            protected_reconstruction=protected_reconstruction,
            mirage_model=mirage_model,
            pgd_loss_history=pgd_loss_history,
            mirage_loss_history=mirage_loss_history,
            epsilon=8 / 255,
        )

        max_perturbation = evaluation["perturbationLinf"]

        # ----------------------------------------------------
        # Images returned directly to the frontend.
        # No server-side image storage is required.
        # ----------------------------------------------------

        protected_url = tensor_to_data_url(
            protected_image
        )

        reconstruction_url = tensor_to_data_url(
            protected_reconstruction
        )

        original_url = tensor_to_data_url(
            image
        )

        # ----------------------------------------------------
        # Downstream open-source model evaluation
        # ----------------------------------------------------
        print("\nRunning Stable Diffusion v1.5 Img2Img evaluator...")
        diffusion_input = tensor_to_pil(protected_image)
        with torch.inference_mode():
            generated_pil = diffusion_pipe(
                prompt="a detailed realistic photograph matching the input image",
                image=diffusion_input,
                strength=0.65,
                guidance_scale=7.5,
                num_inference_steps=20,
            ).images[0]

        downstream_evaluation = evaluate_downstream_reconstruction(
            original_image=image,
            protected_image=protected_image,
            generated_pil=generated_pil,
            device=device,
        )
        downstream_url = tensor_to_data_url(
            pil_to_backend_tensor(generated_pil)
        )

        print("✓ Downstream reconstruction complete")
        print("\n✓ Protection complete")
        print(
            f"Latent distance: {final_latent_distance:.6f}"
        )
        print(
            f"Reconstruction MSE: "
            f"{protected_reconstruction_mse:.6f}"
        )
        print(
            f"Max perturbation: "
            f"{max_perturbation:.6f}"
        )

        return {
            "success": True,

            "originalUrl": original_url,
            "protectedUrl": protected_url,
            "reconstructionUrl": reconstruction_url,

            "latentMean": f"{latent_mean:.6f}",
            "latentStd": f"{latent_std:.6f}",
            "latentShape": str(latent_shape),

            "reconstructionMse": (
                f"{protected_reconstruction_mse:.6f}"
            ),

            "maxPerturbation": (
                f"{max_perturbation:.6f}"
            ),

            "finalLatentDistance": (
                f"{final_latent_distance:.6f}"
            ),

            # Kept as a simple numeric array for the existing
            # frontend chart.
            "lossCurve": [
                float(x)
                for x in pgd_loss_history
            ],

            # Also expose both curves for future UI use.
            "pgdLossCurve": [
                float(x)
                for x in pgd_loss_history
            ],

            "mirageLossCurve": [
                float(x)
                for x in mirage_loss_history
            ],

            # Backend evaluator results.
            "evaluation": evaluation,
            "downstreamEvaluation": downstream_evaluation,
            "aiReconstructionUrl": downstream_url,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("\n❌ /protect failed")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
