# ICLR 2027 full-paper submission — title and abstract (paste into OpenReview)

Matches `paper_assets/iclr27/main.tex` as of 2026-09-26. Titles and abstracts may be edited up to the paper deadline
(Sep 25 AOE = Sep 26 20:00 SGT).

## Title
PaletteDiff: Native-Resolution Pixel-Art Sprite Generation with Palette-Projected Diffusion

## Abstract
Game sprites of 12–24 px are among the smallest images people draw on purpose, and their medium is defined by a
handful of flat colours and hard alpha edges. Despite rapid progress in text-to-image generation, existing systems
render such sprites as shrunken illustrations with soft shading and around a hundred colours. We identify two root
causes: the training targets of sprite corpora are dominated by downscaled artwork rather than native pixel art, and
continuous diffusion has no mechanism to express the few-colour structure of the medium. To address these challenges,
we propose PaletteDiff, a framework for native-resolution sprite generation. First, we introduce Native-Resolution
Target Rebuilding, which removes duplicate leakage, forbids aggressive downscaling and adds 15k permissively licensed
sprites, together with a native evaluation protocol that scores against real low-resolution pixel art only. Second, we
propose Palette-Projected Sampling, a training-free decoding rule that projects the predicted clean image onto a
per-sprite k-colour palette during the late denoising steps, so that the remaining steps repair the quantisation seams.
Third, combined with SDEdit-style re-noising, the same rule yields a Palette Refinement module for sprites from
generators we did not train. PaletteDiff lowers FD-DINOv2 at 16 px from 167.2 to 31.8 ± 0.5; a per-sprite palette is
the decisive ingredient, and imposing it inside the sampler improves on the same quantisation applied afterwards by a
further 18%. Under the native protocol, PaletteDiff has the lowest FD among gpt-image-2, FLUX.2 and SDXL pipelines
given the same palette post-process; Palette Refinement improves all three external systems by 36–41% while
preserving 80% of their silhouettes.

## Notes for the submission form
- Keywords: pixel art, diffusion models, sampling, constrained decoding, evaluation protocol
- TL;DR: Rebuilding sprite targets at native resolution and projecting each sample onto its own small palette during
  late denoising yields sprites 2.5–4.6× closer to real pixel art than strong text-to-image pipelines.
- Primary area: generative models

## Change log vs the registered abstract
- 2026-09-26: rewritten around the named framework (PaletteDiff) and its three components. Removed the "44%" figure,
  which was a v7h-era number; external result now 53.1 ± 2.3 (three seeds) instead of 50.8 (one seed); leakage now
  16.2% instead of 15.5%; removed the uncited "the corpus this area trains on".
- 2026-09-26 (review rounds): removed the "98% backbone gap" claim; "natively small" → "permissively licensed";
  SDEdit credited; post-hoc vs in-sampler gain stated explicitly (41.6 → 34.1, 18%).
