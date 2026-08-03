# Project Presentation Script

## Project Title Options (pick one, or present 2 as "working title")

1. **"NoiseGuard: A Camera-Agnostic Adversarial Perturbation Framework for Preventing Unauthorized AI-Based Image Manipulation"**
2. **"PixelShield: Real-Time Protective Noise Injection Against Generative AI Image Editing"**
3. **"CloakCam: A Lightweight Adversarial Defense Pipeline Against LLM/Diffusion-Based Image Tampering"**

(NoiseGuard is the strongest — it signals both the mechanism (noise/perturbation) and the goal (guard/protection) clearly to a non-specialist committee.)

---

## Speaking Script

### 1. Opening (30 seconds)

"Good [morning/afternoon] sir. My project is titled **NoiseGuard** — a Camera-Agnostic Adversarial Perturbation Framework to prevent unauthorized AI-based image manipulation. In simple terms, I'm building a system that protects a photo the moment it's captured, so that if someone tries to edit or manipulate it using an AI image-editing tool without permission, the AI fails to produce a convincing result."

### 2. The Problem (1 minute)

"Sir, today anyone can take a person's photo and use free AI tools — Stable Diffusion, other generative editors — to alter it: face-swaps, fake context, non-consensual edits, deepfake-style manipulation. This is a real and growing privacy and security problem. Traditional protections like watermarks are visible and easy to remove, and they don't actually stop the AI from editing the image — they only mark ownership after the fact. What's needed is something that stops the manipulation itself, before it can happen."

### 3. The Core Idea (1 minute)

"My approach is based on published research in this area — specifically a technique called adversarial perturbation, used in projects like MIT's PhotoGuard and the University of Chicago's Glaze. The idea is: I add an invisible, carefully calculated noise pattern to the photo. To the human eye, the photo looks completely normal. But to an AI image-editing model, this noise disrupts the internal mathematical representation it uses to understand and edit the image — so when someone tries to run it through an AI editor, the output comes out distorted or unusable, instead of a clean manipulated fake."

### 4. What Makes This Project Different — "Camera Agnostic" (1 minute)

"The 'camera-agnostic' part means this protection isn't tied to one specific camera or app. Instead of modifying a single phone's camera app, I'm building it as a pipeline that works on the image data itself — so it can protect a photo whether it comes from a webcam, a phone, or any other camera source, without needing special hardware for each one. This makes the framework portable and realistic to demonstrate, rather than something locked to one device."

### 5. Technical Approach — Kept Simple for Presentation (1.5 minutes)

"At a technical level, I'm using a frozen AI model — the encoder component from a Stable Diffusion model — as a 'surrogate' target. I generate a small amount of pixel-level noise that pushes this encoder's understanding of the image off-target, using a gradient-based optimization technique. Once I've proven this works, I plan to train a small, fast neural network that learns to generate this protective noise in a single step, so it can run in near real-time instead of taking several seconds per photo.

I'm also addressing a known weakness in this area — the perturbation can be weakened when a photo is compressed, for example when uploaded to WhatsApp or Instagram. So I'm building in robustness against JPEG compression directly into the training process, using a technique that simulates compression during training itself."

### 6. Project Plan / Feasibility (1 minute)

"I've structured this as an 8-phase project over 6 months, sir:

- First, I set up the environment and load a frozen AI encoder as my target.
- Then I build the simplest possible version of the attack on a single image, to prove the core concept works.
- Then I verify it — feed the protected photo into an actual AI editing tool and show that the edit fails, versus an unprotected photo where the edit succeeds.
- Then I test robustness against compression, and improve the technique to survive it.
- Only after the core idea is proven do I move to training a fast neural network version, since that's the highest-effort part.
- Finally, I integrate it into a live capture pipeline — webcam or image import — and prepare a full demonstration with benchmark results.

This ordering means I prove the idea works early, in the first few weeks, rather than risking six months on an unproven concept."

### 7. Honest Limitations (30 seconds — shows maturity, don't skip this)

"I want to be transparent, sir — this is an active, unsolved research area, not a finished solution. If an unprotected copy of a photo already exists elsewhere online, this technique can't retroactively protect it. And very determined attackers using denoising or 'purification' techniques can still weaken the protection. My project is a working proof-of-concept and benchmark study, not a claim of a perfect, unbreakable defense — and I'll present my robustness testing results honestly, including where the protection breaks down."

### 8. Closing (20 seconds)

"In summary, sir, NoiseGuard demonstrates a practical, research-grounded approach to giving people control over their own photos in an era of easily accessible generative AI tools — protecting them at the moment of capture, in a way that isn't tied to specific hardware. I'm happy to walk through the technical architecture in more depth or answer any questions."

---

## Anticipated Questions & Suggested Answers

**Q: "Isn't this already been done by PhotoGuard/Glaze?"**
A: "Yes sir, those are my direct inspiration and I'm building on their published techniques. My contribution is combining their approach with real-time, camera-agnostic deployment and JPEG-compression robustness, and providing an open benchmark comparing protected vs. unprotected images against actual editing attempts."

**Q: "What if someone uses a different AI model than the one you trained against?"**
A: "That's called the transferability problem, sir — it's a known limitation I'll test for directly, by checking whether noise trained against one model still works against a different one, and reporting those results honestly as part of my evaluation."

**Q: "Why 6 months for this?"**
A: "The phased plan front-loads proving the core concept in the first 4-6 weeks, sir, so the majority of the timeline goes into robustness testing, training an efficient version, and building a solid demonstration — rather than research risk."
