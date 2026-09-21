# CVPR 2027 rewrite plan

Deadline 2026-11-16 AoE (registration 11-10, supplementary 11-23). Eight pages excluding references;
the current draft is 21 pages of ICLR format, so roughly 60% of the body has to move or go.

## What the paper claims, in the order a vision reviewer will want it

1. Tiny sprites are a medium, not a resolution: a handful of colours and hard alpha edges.
2. The corpus everyone trains on is not that medium — at 16 px only 7.1% of it is natively small, the
   rest is 25–64 px artwork box-downscaled, with 4,888 exact duplicates and 15.5% hold-out leakage.
   The evaluation reference is drawn from the same pool, so it rewards the defect.
3. Under a reference of real low-resolution sprites the rankings invert, including one of our own
   earlier positive results (cross-resolution self-guidance: +44% biased, nothing native).
4. Fixing the targets is worth 167.2 → 117.9. The rest of the distance is colour: real sprites use a
   median of six opaque colours, every sampling rule we tried emits more than fifty.
5. Projecting the predicted x0 onto a per-sprite k-colour palette during late denoising: 31.8 ± 0.5,
   against 41.6 for quantising finished samples, 120.1 for one global palette, 112.2 for a median
   filter that removes the same dither. The gain is the per-sprite palette, not fewer colours.
6. It is a property of the recipe, not of our network: a 69.7M transformer and the 72.5M UNet differ by
   36.5 native FD under plain CFG and by 0.7 under the projection.
7. It works on generators we did not train: re-noising gpt-image-2 / FLUX.2+LoRA / SDXL+LoRA sprites and
   denoising with the rule improves native FD by 36–41% at 16/20/24 px while keeping ~80% of the
   silhouette and most of the prompt alignment.

## Eight-page body

| § | Content | Pages |
|---|---|---|
| 1 | Intro + teaser figure (real vs corpus-16px vs ours vs a large generator downscaled) | 1.0 |
| 2 | Related work: pixelization, sprite generation, palette/quantised generation, FD protocols | 0.5 |
| 3 | The medium and the protocol: corpus audit table, native reference, leakage control | 1.25 |
| 4 | Method: bucketed pixel-space model (short), palette-projected sampling (the substance) | 1.25 |
| 5 | Experiments: main 16 px table, controls table, k sweep, other resolutions, external table | 2.5 |
| 6 | The rule on other systems: refinement table + qualitative figure | 0.75 |
| 7 | Why the protocols disagree + where the error lives (compressed to the essential paragraphs) | 0.5 |
| 8 | Limitations and conclusion | 0.25 |

## Moves to supplementary

- The full data-pipeline description (cutting, dedup, licence bookkeeping, per-source caps).
- The three data interventions that did not earn a place, including the 20k-vs-80k reversal — keep
  three sentences of it in the body as the methodological caution, the rest in supplementary.
- Quality conditioning and its figure.
- Cross-resolution guidance as a negative result: one sentence in the body, the analysis in supplementary.
- Text-encoder, training-length and capacity ablations.
- The CLIP-cost table and the 32 px results.
- Per-resolution k sweeps beyond the headline number.

## Things that must be added before submission

- **Human study.** Materials are built (`src/v6/build_human_study.py`, 40 prompts × 7 pairs × 2
  questions). Blocked on the user choosing a platform and budget. This is the single most valuable
  addition: the central claim is currently checked only by a reference set we defined.
- **SD-piXL on a subset.** Running; 8.5 h per sprite, so the table will say n≈50 and why.
- **Teaser figure.** Does not exist yet; the qualitative material does.
- Anonymous code link (the one remaining `\todo`).

## Template

CVPR 2027 author kit is not released yet (checked 2026-09-21). Draft against `cvpr-org/author-kit`
for 2026 and swap the `.sty` when 2027 appears; the class options are stable across years
(`\usepackage[review]{cvpr}` for submission).

## Status 2026-09-22

The draft is written end to end and compiles: 7 pages including references, so roughly 6 of the 8
allowed body pages are used. Sections: abstract, intro + teaser, medium and protocol, related work,
method, experiments (main table, two controls, k with a validation split, alignment, backbone swap,
other resolutions, external table), the rule on other systems, discussion and limitations.

**Two pages of slack.** The first things to bring back, in order:
1. The data interventions that did not earn a place, above all the 20k-vs-80k reversal — a short
   version is a real methodological contribution and warns readers off the pilot-sized comparison
   that misled us twice.
2. A qualitative figure for the projection itself (fig_palette.png exists in the ICLR draft).
3. The per-image statistics behind "why the protocols disagree".

Still missing and blocked: the human study, the anonymous code link, SD-piXL on its subset.

## Numbers that left the body (checked 2026-09-22)

`scripts/verify_paper_numbers.py` now checks both drafts and reports which headline numbers the CVPR
version dropped. Five, all deliberate, and all owed to the supplementary:

| Number | Where it was | Must appear in |
|---|---|---|
| 139.2 / 97.1 / 74.5 | 32 px results, one rung above the target range | supplementary |
| 29.0 | clean-subset column of the near-duplicate table | supplementary (body keeps the sentence) |
| 37.1 | object-only reference protocol | supplementary |

Nothing else was lost in the retyping: the other fourteen appear in both drafts and all twenty-nine
verified entries still match the scoring output (`src/v6/verify_against_json.py`).

## Supplementary, 2026-09-22

`paper_assets/cvpr27/supp.tex` is written end to end (3 pages, no TODOs left) and carries everything the
body dropped: corpus construction, the three data interventions with the 20k/80k table, the
cross-resolution negative result with both columns for every sampler, the three network ablations with
the full capacity curve, 32 px, the three alternative reference sets (object-only 37.1, clean subset
29.0, Inception), prompt alignment in full with the two cautions the audit produced, and licences.

All five numbers the body dropped now appear here, so nothing verified was lost in the move.

Remaining before submission: the human study, the anonymous code link, SD-piXL on its subset, and a pass
over the figures at CVPR column width.
