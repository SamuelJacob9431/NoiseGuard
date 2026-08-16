# Image Protection Research — NoiseGuard

Research project exploring latent-space perturbations against image-editing diffusion models.

## Overview

NoiseGuard investigates whether small, targeted perturbations applied to an
image can degrade a diffusion model's ability to faithfully reconstruct or
edit it — without visibly changing the image to a human viewer. The core
mechanism is a **PGD (Projected Gradient Descent) attack** run against a
**frozen VAE's latent space**: the perturbation is optimized to push the
image's latent encoding away from its original position, so that anything
downstream (reconstruction, inpainting, style transfer) that depends on that
latent produces a degraded result.

## Roadmap

- [x] **Phase 0 — Environment**
- [x] **Phase 1 — Frozen VAE**
- [x] **Phase 2 — PGD**
- [ ] **Phase 3 — Demo GUI**
- [ ] **Phase 4 — Verification**
- [ ] **Phase 5 — Robustness**
- [ ] **Phase 6 — JPEG-aware PGD**
- [ ] **Phase 7 — Distilled Generator**
- [ ] **Phase 8 — Integration**
- [ ] **Phase 9 — Evaluation**

---

### Phase 0 — Environment ✅

Set up the reproducible research environment.

- Python environment with PyTorch, `diffusers`, and `transformers` pinned to
  known-compatible versions.
- GPU access confirmed (CUDA-capable device required for practical PGD
  iteration speed).
- Project structure established: `/models`, `/data`, `/experiments`,
  `/results`.
- Baseline sanity check: load a pretrained Stable Diffusion VAE and confirm
  encode → decode round-trips without errors.

**Deliverable:** working environment + a script that loads the VAE and
prints its config (latent channels, downsampling factor, etc).

---

### Phase 1 — Frozen VAE ✅

Establish the fixed encoder/decoder that the attack operates against.

- Loaded a pretrained VAE (e.g. Stable Diffusion's `AutoencoderKL`) and set
  it to `eval()` mode with `requires_grad_(False)` on all parameters — the
  VAE itself is never updated, only the input image is.
- Confirmed latent shape for standard inputs (e.g. a 256×256 RGB image
  encodes to a `[1, 4, 32, 32]` latent).
- Measured baseline reconstruction quality (MSE, visual inspection) on a
  small sample set with **no** perturbation applied, to establish a
  reference point for later comparison.

**Deliverable:** `encode(image) -> latent` and `decode(latent) -> image`
utility functions, plus baseline (unprotected) reconstruction MSE numbers.

---

### Phase 2 — PGD ✅

Implement the core adversarial attack.

- Defined the attack objective: maximize the distance between the original
  image's latent and the perturbed image's latent (an untargeted latent
  attack), subject to an L∞ (or L2) budget on the perturbation so the
  change stays visually imperceptible.
- Implemented the PGD loop: forward pass through the frozen VAE encoder,
  compute loss, backpropagate to the input pixels, step, then project the
  perturbation back inside the allowed budget (clip).
- Logged loss per iteration to confirm the attack is actually converging
  (this is the data the loss-curve chart in the demo GUI visualizes).
- Swept step size and iteration count to find a setting that reliably moves
  the latent without introducing visible artifacts.

**Deliverable:** a `pgd_attack(image, epsilon, steps, alpha) -> protected_image`
function, plus loss-curve logs from representative runs.

---

### Phase 3 — Demo GUI ✅

Build an interactive way to run and inspect the pipeline from Phases 1–3,
for demos and qualitative review without touching a script each time.

- **Desktop app** (PySide6 + Matplotlib): load an image, run the pipeline,
  and see the Original / Reconstruction / Protected images side by side,
  along with live latent statistics, reconstruction MSE, final latent
  distance, and the PGD loss curve as it converges.
- **Web version** (HTML/CSS/JS, no dependencies): the same layout and
  metrics, runnable in a browser for quick sharing or presentation, with a
  single clearly-marked integration point (`runPipeline()`) where the real
  backend from Phases 1–3 plugs in.
- Both interfaces currently ship with placeholder pipeline functions
  (`placeholder_run_pipeline` / `runPipeline()`) that define the expected
  input/output shape, so the GUI layer and the research code can be
  developed and tested independently, then connected.
- Status bar and progress states (idle / running / complete / error) so the
  GUI is usable as a live demo during presentations, not just a static
  mockup.

**Deliverable:** `noiseguard_gui.py` (desktop) and `noiseguard_website.html`
(web), both wired to accept real pipeline output from Phase 2–3 code once
connected.

---

### Phase 4 — Verification

*(not started)*

Confirm the attack actually degrades reconstruction, not just perturbs the
latent numerically.

- Run the *protected* image back through encode → decode and compare the
  result against decoding the *original* image's latent.
- Measure **Reconstruction MSE** (protected vs. original reconstruction)
  and **Final Latent Distance** (L2 distance between original and perturbed
  latents) as the two headline metrics.
- Spot-check results visually: protected images should look unchanged to
  the eye, while their reconstructions should show visible degradation
  (blur, color shift, structural distortion) compared to the unprotected
  baseline.
- Collect latent statistics (mean, std, shape) per run for sanity-checking
  that the attack isn't producing degenerate/out-of-distribution latents.

**Deliverable:** a verification report comparing protected vs. unprotected
reconstruction quality across the sample set, with the metrics above.

### Phase 5 — Robustness

*(not started)*

Test whether the protection survives common image transformations an
attacker or platform might apply before feeding the image to a diffusion
model — resizing, cropping, re-encoding, screenshotting.

---

### Phase 6 — JPEG-aware PGD

*(not started)*

Extend the attack to remain effective after JPEG compression, since most
real-world image pipelines (social media, messaging apps) re-compress
uploads. Likely requires a differentiable JPEG approximation in the attack
loop.

---

### Phase 7 — Distilled Generator

*(not started)*

Explore whether a small distilled/student model can approximate the PGD
attack's effect in a single forward pass, trading some protection strength
for dramatically faster runtime (real-time or near-real-time protection).

---

### Phase 8 — Integration

*(not started)*

Package the verified, robust attack into a usable tool — likely connecting
the Phase 3.5 GUI to the real Phase 1–6 backend, plus a batch/CLI mode for
processing many images at once.

---

### Phase 9 — Evaluation

*(not started)*

Formal evaluation: quantitative metrics across a larger dataset, comparison
against existing tools in this space (e.g. Glaze, PhotoGuard), user study or
qualitative review of visual imperceptibility, and write-up of results.
