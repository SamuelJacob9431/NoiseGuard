NoiseGuard
> **Adversarial image protection against diffusion-model
> reconstruction**
NoiseGuard is a research project exploring whether small, constrained
pixel-level perturbations can disrupt the latent representation used by
image-generation and image-editing models while keeping the protected
image visually close to the original.
The current system combines:
a frozen Stable Diffusion VAE
latent-space PGD
a MIRAGE-style surrogate optimization stage
a FastAPI backend
a browser-based HTML/CSS/JavaScript frontend
PyTorch + CUDA GPU execution
a planned independent open-source vision-model evaluation stage
---
Overview
Many diffusion pipelines first encode an image into a lower-dimensional
latent representation. NoiseGuard attacks this representation indirectly
by optimizing the input pixels.
``` text
Original Image
      |
      v
+---------------------+
| Frozen VAE Encoder  |
+----------+----------+
           |
           v
     Original Latent
           |
           | maximize latent distance
           v
      +---------+
      |   PGD   |
      +----+----+
           |
           v
       PGD Image
           |
           v
      +---------+
      | MIRAGE  |
      +----+----+
           |
           v
     Protected Image
           |
           v
        Evaluation
```
The attack is constrained by an L∞ perturbation budget.
Current demo configuration:
``` text
epsilon = 8 / 255
```
---
Current System
The current implementation separates research code from the web
interface.
``` text
+-------------------------------------------------------+
|                    NoiseGuard Web UI                  |
|                 HTML + CSS + JavaScript              |
+---------------------------+---------------------------+
                            |
                            | POST /protect
                            v
+-------------------------------------------------------+
|                    FastAPI Backend                   |
|     Upload -> preprocessing -> PGD -> MIRAGE        |
+------------------+------------------+----------------+
                   |                  |
                   v                  v
          Stable Diffusion VAE   MIRAGE Surrogate
                   |                  |
                   +--------+---------+
                            v
                     Protected Image
```
The backend has been tested in a Kaggle GPU environment and can be
exposed to the browser through a Cloudflare Quick Tunnel during
development.
> **Deployment note:** A Cloudflare Quick Tunnel and Kaggle session are
> development/demo infrastructure, not production hosting.
---
Research Pipeline
Phase 0 --- Environment
Status: Complete
The project environment contains the components required for the current
attack pipeline:
Python
PyTorch
CUDA
`diffusers`
Stable Diffusion VAE
OpenCLIP / Transformers components for evaluation experiments
FastAPI
Uvicorn
Current development GPU:
``` text
NVIDIA GeForce RTX 3050 Laptop GPU
6 GB VRAM
```
The project structure separates model code, attack stages, outputs, and
the web frontend.
---
Phase 1 --- Frozen VAE
Status: Complete
NoiseGuard uses Stable Diffusion's `AutoencoderKL` as the fixed
encoder/decoder.
The VAE is placed in evaluation mode and its parameters are frozen:
``` python
vae.eval()

for parameter in vae.parameters():
    parameter.requires_grad = False
```
The VAE parameters are never updated. The input image is what
receives gradients during the attack.
For a 256 × 256 RGB input, the current pipeline produces:
``` text
Input:
[1, 3, 256, 256]

Latent:
[1, 4, 32, 32]
```
Core operations:
``` python
latent = encode_image(vae, image)
reconstruction = decode_latent(vae, latent)
```
A baseline encode/decode pass provides a reference reconstruction before
protection is applied.
---
Phase 2 --- Latent-Space PGD
Status: Complete
The core attack uses Projected Gradient Descent.
The objective is to increase the distance between the original and
perturbed latent representations while constraining the pixel-space
perturbation.
Conceptually:
``` text
Original Image
      |
      +-----------------> Original Latent
      |
      v
Adversarial Image
      |
      +-----------------> Adversarial Latent
                                  |
                                  v
                         Latent Distance Loss
```
Each PGD iteration:
enables gradients on the input image
encodes the image through the frozen VAE
computes latent distance
backpropagates to the input pixels
updates the image
projects the perturbation back inside the epsilon budget
Current latent-distance loss:
``` python
loss = torch.mean(
    (original_latent - adversarial_latent) ** 2
)
```
Representative configuration:
``` text
epsilon = 8 / 255
alpha   = 2 / 255
steps   = 50
```
Loss values are recorded so the frontend can visualize optimization
progress.
---
Phase 3 --- MIRAGE + Combined Protection
Status: Implemented
After PGD, NoiseGuard applies a second MIRAGE-style surrogate
optimization stage.
The current design uses one total perturbation budget, rather than
giving PGD and MIRAGE independent budgets.
``` text
Original
    |
    v
   PGD
    |
    v
 PGD Image
    |
    v
 MIRAGE
    |
    v
Final Image
    |
    v
Project into Original +/- epsilon
```
The final result is projected back into the original image's epsilon
neighborhood.
Therefore:
``` text
delta = protected_image - original_image

||delta||∞ <= epsilon
```
This keeps the combined attack inside the intended total perturbation
budget.
---
Phase 3 --- Web Demo
Status: Implemented
The browser frontend is separated into:
``` text
frontend/
├── index.html
├── styles.css
└── script.js
```
The interface provides:
image upload
original image preview
protected image preview
reconstruction output
backend status
latent statistics
reconstruction MSE
maximum perturbation
PGD loss curve
idle / running / complete / error states
The browser communicates with the Python backend through:
``` http
POST /protect
```
The main integration point is:
``` javascript
const API_URL = "YOUR_BACKEND_URL";
```
The frontend sends the image as multipart form data:
``` javascript
const form = new FormData();
form.append("file", imageFile);

const response = await fetch(`${API_URL}/protect`, {
    method: "POST",
    body: form
});
```
---
FastAPI Backend
Current API surface:
``` text
GET  /
GET  /health
POST /protect
```
`GET /`
Returns basic service information.
`GET /health`
Checks that the backend is alive and reports the active device.
Example:
``` json
{
  "status": "ok",
  "device": "cuda"
}
```
`POST /protect`
Accepts an uploaded image and runs the protection pipeline.
The current demo response includes fields such as:
``` json
{
  "success": true,
  "reconstructionUrl": "...",
  "protectedUrl": "...",
  "latentMean": "...",
  "latentStd": "...",
  "latentShape": "...",
  "reconstructionMse": "...",
  "finalLatentDistance": "...",
  "lossCurve": [],
  "maxPerturbation": 0.03125,
  "epsilon": 0.031372549
}
```
Some fields in the rapid-demo backend are currently placeholders and are
intended to be replaced by the complete verification pipeline.
---
Independent Vision-Model Evaluation
Status: Planned / In Progress
NoiseGuard is being extended with an independent open-source
vision-model evaluation stage.
The goal is to evaluate the protected image with a model that was not
used to create the perturbation.
``` text
             +--------------+
             | Original     |
             | Image        |
             +------+-------+
                    |
             +------+------+
             |             |
             v             v
       Vision Model   Vision Model
             |             |
             v             v
       Original Score Protected Score
             |             |
             +------+------+
                    |
                    v
                 Compare
```
An initial candidate is a CLIP-family zero-shot image classifier.
Example candidate labels:
``` text
dog
cat
car
building
person
nature
```
Illustrative output:
Label     Original   Protected
---
dog          98.2%       94.7%
cat           0.8%        1.5%
car           0.3%        0.6%
The numbers above are illustrative only, not measured NoiseGuard
results.
This evaluator is intentionally separate from the PGD/MIRAGE objective.
---
Phase 4 --- Verification
Status: Next
The next major goal is to establish that latent displacement actually
causes downstream degradation.
The experiment should compare unprotected and protected reconstruction
behavior.
Headline measurements:
Reconstruction MSE
Measure reconstruction quality and compare protected versus unprotected
results using a clearly defined baseline.
Final Latent Distance
Measure the distance between the original and protected latent
representations.
Visual inspection
The intended behavior is:
``` text
Protected image
      |
      v
Looks approximately unchanged to a human
```
while downstream processing should show measurable degradation if the
protection is effective.
The verification phase should establish this experimentally rather than
assuming that latent displacement automatically implies downstream
failure.
---
Phase 5 --- Robustness
Status: Planned
Test whether protection survives common transformations:
resizing
cropping
JPEG recompression
screenshotting
format conversion
mild image processing
Conceptually:
``` text
Protected Image
      |
      +---- Original protected
      +---- Resize
      +---- Crop
      +---- JPEG
      +---- Screenshot
                 |
                 v
          Evaluate protection
```
---
Phase 6 --- JPEG-Aware PGD
Status: Planned
Real-world image platforms often recompress uploads.
A future attack variant will incorporate a differentiable approximation
of JPEG compression inside the optimization loop.
``` text
Image
  |
  v
 PGD
  |
  v
Differentiable JPEG
  |
  v
 VAE
  |
  v
 Loss
  |
  +-----------------> PGD
```
The goal is to produce perturbations that remain effective after
compression.
---
Phase 7 --- Distilled Generator
Status: Planned
Iterative PGD is computationally expensive.
A future direction is to train a smaller student model to approximate
the attack in one forward pass.
``` text
              PGD
               |
               v
       Training examples
               |
               v
       +---------------+
       | Student Model |
       +-------+-------+
               |
               v
       One-pass protection
```
The intended tradeoff is:
``` text
PGD
+ potentially stronger optimization
- many iterations

Student
+ much faster inference
- potentially lower protection strength
```
---
Phase 8 --- Integration
Status: Planned
The final system is intended to combine:
verified attack pipeline
web GUI
batch/CLI processing
independent model evaluation
robustness testing
experiment logging
reproducible configuration
Potential architecture:
``` text
                    +-------------+
                    | Web Client  |
                    +------+------+
                           |
                           v
                    +-------------+
                    |   FastAPI   |
                    +------+------+
                           |
              +------------+------------+
              |            |            |
              v            v            v
             VAE/PGD     MIRAGE     Evaluation
              |            |            |
              +------------+------------+
                           |
                           v
                    Experiment Results
```
---
Phase 9 --- Evaluation
Status: Planned
Formal evaluation will measure NoiseGuard over a larger image set.
Potential dimensions:
Attack effectiveness
latent distance
reconstruction degradation
downstream edit degradation
Imperceptibility
visual inspection
pixel-space perturbation
image-quality metrics
Robustness
resize
crop
JPEG
screenshot
format conversion
Baselines
Potential comparisons include existing image-protection approaches such
as:
Glaze
PhotoGuard
Comparisons should use the same dataset, transformations, metrics, and
evaluation protocol wherever practical.
---
Repository Structure
The repository is organized around the research phases:
``` text
NoiseGuard/
|
├── images/
|   └── test_image.jpg
|
├── outputs/
|   └── graphs/
|
├── src/
|   |
|   ├── phase1/
|   |   ├── device.py
|   |   ├── vae.py
|   |   ├── image_utils.py
|   |   ├── output_utils.py
|   |   └── main.py
|   |
|   ├── phase2/
|   |   ├── losses.py
|   |   ├── pgd.py
|   |   └── attack.py
|   |
|   └── phase3/
|       ├── mirage_loss.py
|       ├── mirage.py
|       └── combined_attack.py
|
├── frontend/
|   ├── index.html
|   ├── styles.css
|   └── script.js
|
├── config.py
└── README.md
```
The layout may evolve as verification, evaluation, and deployment are
added.
---
Technology Stack
Component                Technology
---
Language                 Python
Deep learning            PyTorch
Diffusion VAE            Hugging Face Diffusers / `AutoencoderKL`
Attack                   PGD
Surrogate stage          MIRAGE-style optimization
API                      FastAPI
Server                   Uvicorn
Frontend                 HTML / CSS / JavaScript
GPU                      CUDA
Development GPU          NVIDIA RTX 3050 6 GB
Evaluation               CLIP-family / open-source vision models
Development deployment   Kaggle GPU + Cloudflare Quick Tunnel
---
Running the Current Demo
1. Prepare the Python environment
Install the required packages and make sure the environment has access
to a CUDA-capable GPU for practical attack iteration speed.
2. Verify CUDA
``` python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
```
3. Load the VAE
The development setup uses:
``` text
stable-diffusion-v1-5/stable-diffusion-v1-5
```
with the VAE loaded from its `vae` subfolder.
4. Run the protection pipeline
The combined pipeline is:
``` text
PGD -> MIRAGE -> final epsilon projection
```
5. Start FastAPI
The current notebook deployment starts Uvicorn using the app object
directly:
``` python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000
)
```
6. Connect the frontend
Open:
``` text
frontend/script.js
```
and set:
``` javascript
const API_URL = "YOUR_BACKEND_URL";
```
Do not append `/protect`; the JavaScript adds the endpoint.
---
Important Research Notes
Frozen model does not mean zero gradients
The VAE parameters are frozen, but gradients must still flow through the
VAE from the loss back to the input image:
``` text
Input Image
     |
     v
Frozen VAE
     |
     v
Latent Loss
     |
     v
Gradient
     |
     v
Input Pixels
```
Therefore, the attack's encoder pass must remain differentiable with
respect to the input.
The parameters are frozen; the input image is optimized.
---
Perturbation budget
A successful attack should not simply produce an obviously corrupted
image.
The constraint is:
``` text
||delta||∞ <= epsilon
```
where:
``` text
delta = protected_image - original_image
```
Current demo:
``` text
epsilon = 8 / 255
```
The combined pipeline projects the final image back into this budget.
---
Limitations
NoiseGuard is currently a research prototype.
The current implementation does not yet establish that:
latent displacement reliably causes degradation across diffusion
models
protection survives JPEG compression
protection survives resizing/cropping
protection transfers across different VAE implementations
protection transfers across different diffusion models
the current attack is stronger than existing approaches
current demo metrics constitute a complete benchmark
These questions are addressed by the later verification, robustness, and
evaluation phases.
---
Research Questions
How much can a small pixel-space perturbation move an image in VAE
latent space?
Does latent displacement translate into degraded diffusion
reconstruction?
How imperceptible can the perturbation remain?
Does protection survive common image transformations?
Can the protection generalize across downstream models?
Can an iterative attack be distilled into a fast one-pass
generator?
How does NoiseGuard compare with existing image-protection
methods?
---
Visual Documentation
For a research-style repository, keep diagrams and experiment outputs in
a dedicated directory:
``` text
docs/
├── architecture.svg
├── pipeline.gif
├── pgd-loss.png
├── original-vs-protected.png
└── evaluation.png
```
SVG is recommended for large architecture diagrams because it remains
sharp when displayed at different sizes.
---
Adding Images to the README
If the repository contains:
``` text
docs/architecture.png
```
use:
``` markdown
![NoiseGuard architecture](docs/architecture.png)
```
For a centered image with a controlled width:
``` html
<p align="center">
  <img
    src="docs/architecture.png"
    alt="NoiseGuard architecture"
    width="900"
  />
</p>
```
For wide diagrams, the second form usually looks better.
---
Adding GIFs
GIFs can be embedded in exactly the same way:
``` markdown
![NoiseGuard pipeline](docs/pipeline.gif)
```
or:
``` html
<p align="center">
  <img
    src="docs/pipeline.gif"
    alt="NoiseGuard pipeline demonstration"
    width="900"
  />
</p>
```
For NoiseGuard, a short demonstration GIF could show:
``` text
Original
   |
   v
PGD optimization
   |
   v
MIRAGE
   |
   v
Protected image
   |
   v
Independent evaluation
```
A short 5--10 second loop is generally preferable to a very large GIF.
---
Recommended README Visual Layout
A strong portfolio/research README can start with a visual immediately
after the title:
``` markdown
# NoiseGuard

> Adversarial image protection against diffusion-model reconstruction.

<p align="center">
  <img src="docs/pipeline.gif" width="900">
</p>

## Overview

...

<p align="center">
  <img src="docs/architecture.svg" width="900">
</p>

## Results

...
```
Recommended diagrams:
Overall system architecture
PGD mechanism
PGD → MIRAGE combined pipeline
Verification experiment
Independent model evaluation
Robustness experiment
This makes the repository read like a research project rather than
simply a collection of Python scripts.
---
Roadmap
[x] Phase 0 --- Environment
[x] Phase 1 --- Frozen VAE
[x] Phase 2 --- PGD
[x] Phase 3 --- Demo GUI / Web API
[ ] Phase 4 --- Verification
[ ] Phase 5 --- Robustness
[ ] Phase 6 --- JPEG-aware PGD
[ ] Phase 7 --- Distilled Generator
[ ] Phase 8 --- Integration
[ ] Phase 9 --- Evaluation
---
Disclaimer
NoiseGuard is an experimental research project intended to study
adversarial perturbations and the robustness of image-generation/editing
systems.
Results should be interpreted as experimental observations rather than
guarantees of protection against arbitrary image-processing or
generative-model pipelines.
