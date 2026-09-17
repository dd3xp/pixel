# ICLR 2027 — title + abstract, rewritten for the 09-16/17 findings
(abstract registration Sep 18 AoE — the user registers on OpenReview; full paper Sep 25 AoE)

All numbers: FD-DINOv2 against a **native** reference (real sprites that are natively <= R px, no downscaled
artwork), 3,000 held-out captions unless marked n=200. Source: experiment_log.md 09-16 / 09-17.

## Title options
1. **Pixel Art Is a Palette, Not a Downscale: Rebuilding Targets and Decoding for 12–24 px Sprite Generation** (recommended)
2. Four Colours Are Enough: Palette-Projected Diffusion for Tiny Pixel-Art Sprites
3. What Your Sprite Dataset Is Actually Teaching You

## Abstract (~230 words)
Text-to-image models fail at the smallest images people draw on purpose: 12–24 px RGBA game sprites, where a
handful of colours and hard alpha edges *are* the medium. We show that the usual failure is not a modelling
failure but a target failure. In the standard sprite corpus only 7% of images are natively 16 px or smaller;
the rest is 25–64 px artwork box-downscaled into the low-resolution buckets, so a model trained on it learns
soft, 50-colour blobs — and the evaluation reference inherits the same bias, rewarding exactly that. We rebuild
the targets (drop 4,888 pixel-identical duplicates that leak 15.5% of the held-out set into training, forbid
downscaling beyond 1.5x, add 14k natively small sprites from three permissive corpora) and we evaluate against
a reference drawn only from real low-resolution pixel art. Under this protocol the data fix alone takes
FD-DINOv2 from 209.6 to 117.9 at 16 px. We then close the remaining gap where it actually is: real sprites use
a median of 6 opaque colours while every sampling configuration we tested emits over 50. Projecting the
predicted x0 onto a per-sprite k-colour palette during the late denoising steps — so the remaining steps repair
the quantisation seams — reaches 31.8 ± 0.5, better than quantising the finished samples (41.6) and 2.6x better
than the strongest external system under the same palette post-process (gpt-image-2, 134.4 vs our 50.8 at
n=200); it also beats the mixed-provenance "real" sprites the old protocol treated as ground truth (77.0).
We report an equally instructive negative result: cross-resolution self-guidance and its internalised one-pass
variant, which win by 40% under the downscale-biased reference, provide no gain once the reference is real
pixel art, while independent-condition guidance transfers and composes with palette projection.

## Key numbers (16 px unless noted, native reference, n=3000)
- data only: v7r 167.2 (old policy, best CFG) -> v8f2 123.1 -> v8n 117.9 (CFG 1.5).
- guidance on the rebuilt model: CFG 1.0 113.2 / CFG 1.5 117.9 / cross-resolution 114.9 / composed 117.0 /
  1b internalised 170.3–178.8 (harmful) / **ICG w1.5 87.2, w2 87.1**.
- palette projection (ICG w2): k=8 50.6(post-hoc) · in-sampler k=6 36.9 · **k=4 31.8 ± 0.5 (3 seeds)** ·
  k=3 38.1 · per-image k from the corpus histogram 39.9 · CFG+k=6 58.4 (guidance and palette compose).
- other resolutions (best k): 12 px 156.5 (k=4) · 20 px 96.7 (k=8) · 24 px 124.2 (k=6).
- external, n=200, same native reference, all with the same 4-colour post-process where applicable:
  ours 50.8 · real mixed-source sprites 77.0 · gpt-image-2 134.4 · FLUX.2-klein+pixel LoRA 150.1 ·
  SDXL+Pixel Art XL 244.2 (unquantised: 284.4 / 318.9 / 436.4).
- palette statistics of real sprites: median 6 colours (12 px: 5, 20/24 px: 7), 22.6% use <= 4.
- old (downscale-biased) protocol, for the appendix: v7r 1b+ICG 6.24 vs v8n CFG2 12.16 — the ranking inverts.
