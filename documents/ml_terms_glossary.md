# ML Terms Glossary — NoiseGuard Project Defense Prep

Organized in three groups: basics, adversarial-attack specific terms, and engineering terms. Each entry has a plain-language explanation and a one-line "how to say it to sir" version.

---

## Group 1: Foundational Terms

### Neural Network
A system made of layers of simple math units ("neurons") that learns patterns from data by adjusting internal numbers called weights. It takes an input (like an image), passes it through these layers, and produces an output (like a classification, or in your case, a transformed version of the image).
**Say it as:** "A neural network is a layered mathematical function that transforms input data using learned weights."

### Weights / Parameters
The internal numbers a neural network adjusts during training to get better at its task. In your project, you **freeze** these — you don't change them at all.
**Say it as:** "Weights are the network's learned internal values. I don't train them — I keep them frozen and use the network only as a fixed target."

### Encoder
A part of a network that converts raw input (an image) into a compressed numeric representation — a list of numbers that captures the "meaning" of the image in a form the rest of the AI system can work with.
**Say it as:** "The encoder converts an image into an internal numeric representation that the AI model uses to understand and later reconstruct or edit it."

### Latent Space / Latent Representation
The compressed numeric form the encoder produces. Not human-readable — just a grid of numbers — but it's what the AI actually "thinks" about your image, instead of thinking about raw pixels.
**Say it as:** "Latent space is the AI's internal, compressed understanding of the image — my attack targets this, not the pixels directly."

### VAE (Variational Autoencoder)
A specific type of encoder-decoder network. The encoder compresses the image into latent space; the decoder reconstructs an image back out of that latent space. Stable Diffusion uses a VAE as its first step.
**Say it as:** "A VAE is the encode-decode component of Stable Diffusion. I attack its encoder half."

### Diffusion Model
The type of AI model behind tools like Stable Diffusion. It generates or edits images by starting from noise (or a real image) and gradually refining it step-by-step into a final image, guided by a text prompt.
**Say it as:** "A diffusion model generates or edits images through a gradual step-by-step refinement process, starting from an encoded representation of the input."

### Training vs. Inference
Training = the process of a model learning by adjusting its weights on lots of data (slow, needs lots of compute). Inference = using an already-trained model to process one input and get an output (fast).
**Say it as:** "Training builds the model's knowledge; inference is just using that already-built knowledge on a new image. My attack happens at inference time — I never train the target model."

---

## Group 2: Adversarial Attack Terms (the heart of your project)

### Adversarial Perturbation / Adversarial Example
A small, deliberately calculated change to an input (like a photo) that causes an AI model to behave incorrectly, while looking unchanged or nearly unchanged to a human.
**Say it as:** "An adversarial perturbation is a small, targeted change to the image that causes the AI model to misinterpret it, while remaining invisible to a human viewer."

### Gradient
A mathematical measure of "which direction, and how much, would change the output if I nudged this input slightly." Neural networks are built so this can be calculated automatically.
**Say it as:** "The gradient tells me, for each pixel, which direction to change it in to push the model's output toward my target."

### Backpropagation
The technique used to calculate gradients through a neural network, normally used to update a model's weights during training. In your project, you use the same math, but instead of adjusting weights, you adjust the input image.
**Say it as:** "Backpropagation is normally used to train weights — I repurpose it to adjust the image pixels instead, while keeping weights frozen."

### PGD (Projected Gradient Descent)
The specific attack method: repeatedly (1) compute the gradient, (2) nudge the image slightly in that direction, (3) clip/"project" the total change back within a small allowed limit, so the change stays bounded and invisible. Repeat for many steps.
**Say it as:** "PGD is an iterative method — small gradient-based nudges to the image, repeated many times, with the total change kept within a strict, invisible limit."

### Epsilon (ε)
The maximum allowed size of the perturbation — the "budget" for how much any pixel is allowed to change. A small epsilon keeps the change invisible; too large an epsilon and the noise becomes visible.
**Say it as:** "Epsilon is the perturbation budget — how far pixels are allowed to shift. I keep it small enough to stay imperceptible."

### Target / Target Representation
What you're pushing the image's latent representation *toward* — commonly random noise or an unrelated image, so the AI's "understanding" of your photo becomes meaningless.
**Say it as:** "I optimize the noise so the encoder's output for my photo moves toward a meaningless target, instead of the photo's real representation."

### Surrogate Model
The specific AI model you attack directly, standing in for the broader category of models an attacker might actually use (since you can't access every possible tool an attacker might use).
**Say it as:** "I use a surrogate model — a real, publicly available encoder — as a stand-in, since I can't attack every possible AI tool directly."

### Transferability
Whether a perturbation crafted against one model (your surrogate) also works against a *different* model the attacker might actually use. This is a known open challenge in this field.
**Say it as:** "Transferability is whether my protection, trained against one model, still works against a different one — I test this directly and report it honestly as a limitation."

### White-box vs. Black-box Attack
White-box = you have full access to the target model's internals (weights, gradients) — this is what you're doing, since you load the real VAE yourself. Black-box = you don't have access and can only observe outputs.
**Say it as:** "My attack is white-box — I have direct access to the surrogate model's internals to compute exact gradients."

---

## Group 3: Robustness & Engineering Terms

### Purification / Denoising Attack
A counter-attack where someone tries to remove your protective noise by blurring, denoising, or otherwise "cleaning" the image before feeding it to their AI editor.
**Say it as:** "Purification attacks try to wash out my protective noise using blur or denoising before editing — I test robustness against this directly."

### Differentiable JPEG
A version of the JPEG compression process rewritten so gradients can flow through it during training. Normal JPEG compression involves a rounding step that blocks gradient calculation; this technique approximates that step so it doesn't.
**Say it as:** "Differentiable JPEG lets me include compression in my training loop, so my noise is optimized to survive compression rather than just the raw encoder attack."

### Distillation (in your project's context)
Training a small, fast neural network to reproduce the output of a slow, iterative process (your PGD loop) in a single fast step, so it can run in near real-time.
**Say it as:** "I'm distilling the slow iterative attack into a fast single-pass network, so protection can be generated in real time at capture."

### Mixed Precision (fp16)
Running calculations using 16-bit numbers instead of the default 32-bit, which roughly halves memory use and speeds up computation, with a small, usually acceptable accuracy tradeoff.
**Say it as:** "I use fp16 mixed precision to fit the model in my GPU's memory and speed up computation."

### Gradient Checkpointing
A memory-saving technique where, instead of storing every intermediate calculation during a forward pass for later use in backpropagation, the network recomputes some of them on demand — trading extra compute time for lower memory use.
**Say it as:** "Gradient checkpointing trades some extra computation time for significantly lower memory usage, which is necessary given my GPU's memory limit."

### Frozen Model
A model whose weights are locked — you use it to compute outputs and gradients, but never update its internal values.
**Say it as:** "The surrogate model stays frozen throughout — I only ever modify the input image, never the model."

### Batch Size
How many images are processed together in one step. Smaller batch size uses less memory but can make training slower or noisier.
**Say it as:** "I use a small batch size to fit within my GPU's memory limits."

---

## Quick-Reference: One-Line Answers for Common "Explain X" Prompts

| If sir asks... | Say... |
|---|---|
| "What is the encoder?" | "The component that converts the image into the AI's internal numeric understanding of it." |
| "How does the noise get generated?" | "Through gradient-based optimization — repeatedly nudging pixels to push the encoder's output toward a meaningless target." |
| "Why is it invisible?" | "The total pixel change is bounded by a small epsilon value during optimization." |
| "What is PGD?" | "Projected Gradient Descent — the iterative method I use to generate the noise." |
| "What's the biggest limitation?" | "Transferability to other models, and vulnerability to purification/denoising attacks — both of which I test and report on." |
| "Why Stable Diffusion's VAE specifically?" | "Because it's the actual component real AI image-editing tools use, so results are meaningful, not just theoretical." |
