from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"

IMAGE_DIR = PROJECT_ROOT / "images"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = PROJECT_ROOT / "models"

GRAPH_DIR = OUTPUT_DIR / "graphs"

LOG_DIR = OUTPUT_DIR / "logs"

# ==========================================================
# Files
# ==========================================================

TEST_IMAGE = IMAGE_DIR / "test_image.jpg"

RECONSTRUCTION_IMAGE = OUTPUT_DIR / "reconstructed.jpg"

PROTECTED_IMAGE = OUTPUT_DIR / "protected.jpg"

LOSS_CURVE = GRAPH_DIR / "loss_curve.png"

LATENT_PLOT = GRAPH_DIR / "latent_plot.png"

# ==========================================================
# Model
# ==========================================================

MODEL_ID = "runwayml/stable-diffusion-v1-5"

IMAGE_SIZE = 256

# ==========================================================
# Create folders automatically
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)