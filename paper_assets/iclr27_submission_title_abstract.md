# ICLR 2027 abstract registration — final text (paste into OpenReview)

## Title
Pixel Art Is a Palette, Not a Downscale: Rebuilding Targets and Decoding for 12–24 px Sprite Generation

## Abstract (238 words)
Text-to-image models fail at the smallest images people draw on purpose: 12–24 px RGBA game sprites, where a handful
of colours and hard alpha edges are the medium. We show that the usual failure is not a modelling failure but a target
failure. In the sprite corpus this area trains on, only 7.1% of the images are natively 16 px or smaller; the rest is
25–64 px artwork box-downscaled into the low-resolution buckets, and 4,888 pixel-identical duplicates leave a twin in
training for 15.5% of the held-out set. A model fitted to those targets learns soft, fifty-colour blobs, and the
evaluation reference, drawn from the same pool, rewards exactly that. We rebuild the targets and evaluate against
references that contain only real low-resolution pixel art. Under this protocol the data work alone takes FD-DINOv2
from 167.2 to 117.9 at 16 px. We then close the remaining gap where it actually lies: real sprites use a median of six
opaque colours while every sampling rule we tested emits more than fifty. Projecting the predicted x0 onto a per-sprite
k-colour palette during the late denoising steps reaches 31.8 ± 0.5 — better than quantising finished samples (41.6),
than one corpus-wide palette (120.1) and than a median filter that removes the same dither (112.2). Against external
systems given the same palette post-process we reach 50.8 versus 134.4 for gpt-image-2 with downscaling. We also report
a negative result: cross-resolution self-guidance, which wins by 40% under the downscale-biased reference, gives
nothing under a native one.

## Notes for the submission form
- Keywords: pixel art, diffusion models, sampling, dataset quality, evaluation protocol
- TL;DR: For tiny sprites the targets and the reference set decide the result; projecting each sample onto its own
  small palette during late denoising closes most of the remaining gap.
- Primary area: generative models / datasets and benchmarks

## Change log vs the registered abstract
- 2026-09-19: the data-only gain now reads 167.2 -> 117.9 (both at 80k steps, best-tuned CFG). The registered
  text said 209.6 -> 117.9, which compared a 20k control against an 80k model. Update the abstract field when
  submitting the full paper.
