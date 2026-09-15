# ICLR 2027 — title and abstract draft (for OpenReview abstract registration, deadline Sep 18 AoE)

All numbers: matched FD-DINOv2 on 3,000 held-out captions, recaptioned model (v7r), mean over 3 seeds unless noted.
Sources: pixel_art_research_20260816/experiment_log.md (09-14 .. 09-16 entries).

## Title options
1. **Your Lower Resolution Is Your Best Weak Model: Cross-Resolution Guidance for Tiny Pixel-Art Diffusion** (recommended)
2. Cross-Resolution Guidance, Internalized: One-Pass Generation of 12–24 px Pixel-Art Sprites
3. Guiding Tiny Sprites with Their Own Lower Resolution

## Abstract (≈230 words)

Text-to-image models fail at the smallest images people draw on purpose: 12–24 px RGBA game sprites, where every
pixel is a design decision and a large generator followed by downscaling or learned pixelization blurs exactly those
decisions. We train a pixel-space diffusion model over a ladder of resolution buckets and show that it contains its own
best weak model: the same network queried under a *lower* resolution label (optionally through an early snapshot)
yields a structure-aligned, lower-contrast prediction whose difference from the target-resolution prediction is a
precise guidance direction. The effect is directional — a higher or a random bucket hurts — and it is not a
denoising-error correction: the MSE-optimal extrapolation weight stays at 0.83–1.07 on both forward-noised and
self-rolled states. We then internalize this reference with a guidance-free-training objective, so that a single
forward pass reproduces and surpasses the two-pass rule without storing any extra weights; with an empty-caption or a
random-condition reference, the identical recipe is 39% and 10% worse. At 16 px the internalized model lowers FD from
11.28 (best-tuned CFG) to 7.23 at one network evaluation, and to 6.25 when combined with independent-condition
guidance, 16% below the strongest training-free baseline; it leads at 20 and 24 px as well (−30%, −34%), and an 8 px
rung extends the rule to 12 px. Generating with gpt-image-2 or nano-banana-2 and then downscaling, applying PixelOE,
or applying learned pixelization gives a 3–7× higher FD at 16 px. We also report where the idea does not transfer: SDXL's
original-size conditioning is not a usable reference.

## Numbers behind each claim (for checking before submission)
- best CFG 16 px 11.28 ± 0.54; 1b 7.23 ± 0.45 (re-training 7.20 ± 0.28); 1b + ICG 6.25 ± 0.09 (re-training 6.31 ± 0.33); ICG 7.46 ± 0.37.
- same recipe, empty-caption reference (plain GFT) 10.05 ± 0.40 → (10.05 − 7.23)/7.23 = 39% worse; ICG reference 7.96 ± 0.22 → 10% worse.
- 20 px: CFG 34.21, 1b 24.06 (−30%); 24 px: CFG 65.87, 1b 43.35 (−34%, 2 seeds; third seed to add).
- 12 px (v7r8, 8 px rung): label ref 7.50 vs CFG 8.40 (seed 0; 1b-v7r8 pending) — keep the claim qualitative until 1b-v7r8 is in.
- external, n = 200 protocol, 16 px: ours 27.81 vs floor 26.53; gpt-image-2 downscale 82.23, nano 87.35, PixelOE 146–151,
  Make Your Own Sprites 155–187 → distance to floor: ours 1.3 vs 55.7–160 ("3–6×" is conservative only if phrased as FD ratio 82/27.8 = 3.0 .. 187/27.8 = 6.7).
- directionality: bucket 20/24/64 references worse than best CFG; random bucket (ICG-label) 12.23 > 11.10.
- w*: T0 0.83–1.07, T1 max 0.82–1.07 across 16/20/24 px.
- SDXL: orig-512 reference 644.13 vs CFG 644.18; orig-256 658.98 (worse).
