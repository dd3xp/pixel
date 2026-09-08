# Architecture scouting pass — 2026-09-09

Scope: literature-and-ideas pass for a **new architecture / new training objective** that could plausibly
get matched FD-DINOv2@native below **7.53** on 12/16/20/24/32 px RGBA sprites, trainable in ~4 h on one A100.
Explicitly **not** sampling-time guidance. Every arXiv citation below was verified by fetching the abstract
page on 2026-09-09; anything unverified is flagged.

---

## 0. Two facts I checked in the repo that should change how the shortlist is read

These are not literature findings, but they invalidate or re-rank several of the obvious candidates, so
they go first.

### 0.1 The FD@16 target is **92 % box-downscaled art**, not native pixel art

`src/v6/fd_fair.py::real_split()` takes `data/oga_clean/**.png`, seed-0 shuffle, first 3000 as reference, and
pushes each through `train_cond.to_tensor(16)`. I measured the native size distribution of exactly that
reference set:

| native max side | ≤16 | 17–24 | 25–32 | 33–48 | 49–64 | >64 |
|---|---|---|---|---|---|---|
| count (of 3000) | **230** | 780 | 559 | 905 | 431 | 95 |

So **only 7.7 %** of the "real 16 px sprites" the metric compares against were drawn at ≤16 px. The other
92 % are 24–64 px artwork put through a premultiplied BOX downscale. That is exactly why the reference has
34 colours / 20 % flat while true native-16 sprites have 6 colours / 41 % flat.

Consequences:

- **"Exactly-flat regions and hard edges as a property of the parameterisation" is aiming at the wrong
  target.** A model that is flat-by-construction overshoots the reference. This retro-explains almost the
  whole falsified list in one line: 8-colour octree 65.03, palette head (7 colours/sprite) 78.34,
  ordered-palette discrete 252.3, TV w=1.0 569.35 — every mechanism that *increased* pixel-art-ness past
  the 15–20-colour sweet spot blew up, and the one that stopped at 16 colours (35.24) was the best free
  baseline. The metric wants *moderately* quantised, box-downscale-looking output.
- Any new architecture with a **hard colour budget is likely dead on arrival**, independent of how well it
  is implemented. Vocabulary must be large (≥ a few thousand effective colours), or continuous.
- The framing "natively at 12/16/20/24/32 px, not downscaled from big images" in the project brief is not
  what the training pipeline (`src/v6/train_v7.py`, `LOW=[12,16,20,24]`, `Image.BOX`) or the metric actually
  do. Worth fixing in the paper before a reviewer finds it.

### 0.2 The corpus is a mixture over an *unobserved* downscale factor, and the model is forced to average over it

At bucket 16 the model sees, under one label, both a crisp native-16 sprite (6 colours, 41 % flat) and a
64 px sprite box-downscaled by 4 (antialiased, many colours, ~0 % flat). The bucket embedding tells the model
the *output* size, never the *source* size. Averaging over a genuine 5-way mixture is textbook mean
regression, and its signature is precisely what `stats_simplicity.py` reports for v7h: 73 unique colours vs
34 real, flat fraction 0.03 vs 0.20. This is candidate **A2** below, and it is also the most economical
explanation of *why the existing 7.53 trick works*: guiding away from the model's belief under a lower
bucket label is a crude sampling-time proxy for a conditioning variable the model was never given.

---

## 1. Ranked shortlist

Ranking is by (expected FD gain) × (probability the mechanism is real), with novelty as a tiebreak, not as
the primary key. Novelty verdicts are in §2.

---

### A1 — Masked discrete diffusion over **per-channel 256-level raw-pixel tokens** (large vocabulary, neighbour-aware CE, editable decoding)

**Mechanism.** Treat a 16 px RGBA sprite as 4 × 256 = 1024 tokens, each a 256-way categorical over the actual
8-bit channel value — no palette, no VQ, no tokenizer, so the quantisation floor is exactly zero (the assets
*are* 8-bit). Train the continuous-time masked-diffusion ELBO (a weighted integral of cross-entropies, MD4),
with a small bidirectional transformer, CLIP cross-attention, bucket as a token. Decode by confidence-ordered
unmasking, plus a token-editing pass so already-unmasked values can be revised. Two fixes are load-bearing:
(i) the vocabulary is per-channel 8-bit, not a global palette; (ii) the CE gives positive signal to values
*near* the ground truth in embedding space (Nemotron's Grouped Cross-Entropy), so "which of two adjacent
colours" is a graded rather than a 0/1 decision.

**Closest prior work (verified).**
- Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, Michalis K. Titsias. *Simplified and Generalized Masked
  Diffusion for Discrete Data.* NeurIPS 2024. arXiv:2406.04329. — the abstract explicitly reports
  **pixel-level** image modelling at 2.75 bpd (CIFAR-10) and 3.40 bpd (ImageNet 64×64), "better than
  autoregressive models of similar sizes". This is the existence proof that masked diffusion over raw
  256-valued pixel tokens is a working recipe, not a toy.
- Shufan Li, Greg Heinrich, Hanrong Ye, Yonggan Fu, Aditya Grover, Jan Kautz, Pavlo Molchanov.
  *Nemotron-Labs-Diffusion-Image: Advancing Masked Discrete Diffusion for High-Resolution Image Synthesis.*
  arXiv:2606.29814 (Jun 2026). — names exactly the two MDM pathologies that killed v6f: no self-correction
  after unmasking, and *sparse per-token training signal as vocabulary grows*; fixes them with token editing
  and Grouped Cross-Entropy. Operates on VQ latents, **not** raw pixels.
- Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman. *MaskGIT: Masked Generative Image
  Transformer.* arXiv:2202.04200 (CVPR 2022). — confidence-ordered parallel decoding.
- Patrick Pynadath, Jiaxin Shi, Ruqi Zhang. *CANDI: Hybrid Discrete-Continuous Diffusion Models.*
  arXiv:2510.22510 (Oct 2025). — the hybrid variant, if pure MDM stalls at low NFE.

**Why it could beat 7.53 here.** Two distinct mechanisms, both about *structure*, not just flatness.
(a) The reverse step of a continuous DDPM is a per-pixel **independent** Gaussian; all cross-pixel structure
has to be smuggled through the mean. At 256 pixels with a strong global silhouette constraint, this
independence error is exactly the "can't commit" failure the project has been fighting from the guidance
side for three cycles. Masked/AR decoding has no such factorisation error — the chain rule is exact, and each
unmasked pixel is a *committed* value that later pixels condition on. (b) The falsified energy-score probe
(#5) was an attempt at the same fix that failed *because it sampled per-step in a continuous space and left
speckle*; here every decision is a discrete commitment conditioned on all previous commitments, so there is
no per-step noise injection to speckle.
Crucially, **the previous kill of this family is confounded.** From `experiment_log.md` line 473: v6f defined
the diffusion on the **32 indices of the DawnBringer32 palette** — a *global corpus-wide* 32-colour vocabulary,
when the reference's own median is 34 colours *per sprite*. The recorded failure analysis ("gradients must be
built from adjacent palette indices; K-way CE barely distinguishes which nearby colour; the model degenerates
to one dominant colour per region") is verbatim the sparse-signal pathology GCE was invented for. v_ord (252.3)
inherited the same 32-entry vocabulary and added a hand-built 1-D OKLab ordering on top. Neither run tests
"discrete at low resolution"; both test "a 32-colour global palette", which §0.1 says is hopeless on its own.

**Cheapest falsifying probe (order matters — the first step costs zero GPU).**
1. **Zero-training vocabulary-floor check, ~10 min.** Push the 3000 held-out real sprites through the proposed
   discretisation and score `fd_fair` against the unquantised reference. For per-channel 8-bit this must come
   back ≈3.45 (it is the identity). Then repeat for any palette variant you are tempted by: DB32 → expect a
   number in the tens; a 4096-colour corpus palette → expect <4. **If the floor of a variant exceeds ~4.5,
   that variant cannot reach 7.53 and is dead before training.** This one check would have killed v6f and
   v_ord in ten minutes.
2. Train the 1024-token MDM, ~72 M params, 40 k steps (≈4 h), same data/text encoder/buckets as v7h.
3. Sample 3000 matched captions, `fd_fair --size 16`, plus `fd_decomp` and `stats_simplicity`.

**Decision number:** bare (no guidance) matched FD ≤ 12 → the mechanism is real, proceed to add guidance and
compare against 7.53. 12–20 → ambiguous, one retry with GCE width and decoding-step count swept. ≥ 20 → the
discrete family is genuinely dead at this resolution and can be closed for good with a clean explanation.

**Biggest reason it might fail.** Per-channel factorisation. Splitting a pixel into four independent 256-way
categoricals reintroduces exactly the independence assumption I criticised, one level down: R, G, B of the
same pixel are strongly correlated, and independent argmaxes produce off-manifold colours that FD-DINOv2
punishes hard, since each pixel is one patch. PixelCNN++ (Tim Salimans, Andrej Karpathy, Xi Chen, Diederik P.
Kingma, arXiv:1701.05517) hit this and moved to conditioning on **whole pixels** rather than R/G/B sub-pixels;
the mitigation here is the same — order the four channel tokens of a pixel adjacently and let the model see
already-decoded channels of the same pixel.

---

### A2 — Source-scale conditioning: model the downscale factor as an observed (or latent) variable

**Mechanism.** Every training sample currently carries one conditioning variable, the target bucket R. Add a
second: the **native size of the source asset** (equivalently the downscale factor s/R), as an embedding
alongside the bucket embedding, with classifier-free dropout. The model then represents
p(x | text, R, s) instead of being forced to represent the mixture ∫ p(x | text, R, s) p(s) ds by a single
mean. At sampling, either draw s from the corpus marginal to match the reference distribution exactly, or fix
s to expose a crisp-native mode. The stronger version treats s as a **latent** and trains a small prior over
it, which lets you marginalise correctly at test time.

**Closest prior work (verified).** Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn,
Jonas Müller, Joe Penna, Robin Rombach. *SDXL: Improving Latent Diffusion Models for High-Resolution Image
Synthesis.* arXiv:2307.01952 (Jul 2023). The abstract claims "multiple novel conditioning schemes"; the
original-image-size micro-conditioning is in the body, **not** the abstract — *flagged: I verified the paper
and the claim of novel conditioning schemes, but could not verify the size-conditioning detail from the
abstract page.* Also relevant and verified: Ruozhen He, Moayed Haji-Ali, Ziyan Yang, Vicente Ordonez.
*NoiseShift: Resolution-Aware Noise Recalibration for Better Low-Resolution Image Generation.*
arXiv:2510.02307 (Oct 2025; rev. May 2026) — same diagnosis one level over (resolution mismatch mis-calibrates
the denoiser), fixed by re-indexing noise conditioning rather than by adding a conditioning variable.

**Why it could beat 7.53 here.** It is the only candidate on this list whose predicted symptom *already
matches the measured one*. Mean-over-mixture predicts too many colours, too little flat area, and depressed
local contrast; v7h measures 73 colours / 0.03 flat against 34 / 0.20. And the project's own best result is a
sampling-time hack that manipulates the *other* resolution label — i.e. the field already knows this axis
carries the signal; A2 puts it in the model where it belongs. Expected effect is large because it removes a
bias rather than adding a prior.

**Cheapest falsifying probe.** Fine-tune v7h for 20 k steps (≈1.5 h) with the extra scale embedding
(the native size is free — it is `Image.open(p).size` at dataset build time). Evaluate three ways: s sampled
from the corpus marginal, s fixed to "native", s fixed to "heavily downscaled". **Decision number:** the
corpus-marginal setting must beat v7h bare (21.98) by ≥4 points *and* the three settings must separate by
≥4 points from each other (that separation is the evidence that a real mixture was being averaged). If the
three settings are indistinguishable, the mixture hypothesis is false and A2 dies, along with a chunk of the
current mechanistic story for the 7.53 result.

**Biggest reason it might fail.** The bucket embedding may already be carrying the information implicitly
(a 12 px sample is *usually* a heavy downscale), in which case the new variable is redundant and you get the
`probe_selfq` outcome — a plausible-looking small delta that dissolves under a paired control. Run the paired
control from the start.

---

### A3 — Masked autoregressive over 256 pixel tokens with a **copy-or-create head** (continuous, no vocabulary)

**Mechanism.** MAR backbone: a bidirectional transformer over 256 RGBA pixel tokens, random-order masked
autoregressive decoding, with a small per-token head that models p(pixel | context) — but instead of MAR's
per-token diffusion head, the head is a **mixture between a pointer over already-decoded pixels of the same
image and a "create" component that emits a genuinely new colour**, with a learned soft switch. Copying is
exact: the emitted RGBA is bit-identical to the pointed-at pixel. Flat regions, hard edges and per-image
palettes then emerge as literal repetition, with no fixed K, no palette stage, no post-hoc quantisation, and
no vocabulary quantisation floor. The number of distinct colours in a sprite is an *output* of the model, not
a hyperparameter.

**Closest prior work (verified).**
- Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, Kaiming He. *Autoregressive Image Generation without
  Vector Quantization.* NeurIPS 2024 Spotlight. arXiv:2406.11838. — the backbone and the idea of a
  per-token distribution head over continuous values.
- Abigail See, Peter J. Liu, Christopher D. Manning. *Get To The Point: Summarization with Pointer-Generator
  Networks.* arXiv:1704.04368 (Apr 2017). — the copy/generate soft switch. *Flagged: the arXiv abstract page
  does not state a venue; this is ACL 2017 by common knowledge, unverified from the page.*
- Searched specifically for a copy/pointer head in image generation ("copy mechanism image generation reuse
  previously generated pixel colour", "pointer head predict colour equal to neighbouring pixel"): **no hit.**
  Nearest things in the image literature are Infinity's bitwise infinite-vocabulary classifier (Jian Han,
  Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, Xiaobing Liu. *Infinity: Scaling
  Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis.* arXiv:2412.04431) — which also escapes
  a fixed vocabulary, but by bit factorisation rather than by reuse.

**Why it could beat 7.53 here.** It is the only design on this list that produces exact colour reuse
*without* a colour budget, which is the exact needle §0.1 says you have to thread: the target has ~34 colours
and 20 % flat area, so you need a mechanism that can be flat *sometimes* and antialiased *sometimes*, chosen
per pixel. A hard palette can't; a continuous denoiser won't; a copy head can, and the switch probability is
directly interpretable and directly comparable to the measured flat-region fraction.

**Cheapest falsifying probe.** Two-stage, and the first stage is cheap.
1. **Teacher-forced reconstruction probe, ~30 min.** Train only the head on ground-truth contexts (no
   generation): does the copy switch learn to fire on ≈20 % of pixels at bucket 16 and ≈41 % on native-16
   sprites? If the switch collapses to always-create or always-copy, the parameterisation has no purchase and
   you stop here.
2. Full 40 k-step train, sample, `fd_fair`. **Decision number:** bare matched FD ≤ 12, same as A1, plus the
   flat-region fraction of samples must land in [0.12, 0.28] without any post-processing.

**Biggest reason it might fail.** Random-order masked decoding on 256 tokens with a copy head may just learn
a very good *inpainter* and a mediocre *generator*: the first few tokens, decided with almost no context, set
the silhouette, and copying cannot help there. Since the project's own diagnosis is that the residual
difficulty is structure rather than colour, a mechanism whose whole advantage is colour reuse may buy the
flat-region statistic and leave FD roughly where it was.

---

### A4 — Objective overhaul: **x0-prediction + resolution-calibrated logSNR** on a plain pixel transformer

**Mechanism.** Keep the data and the text encoder; change three things at once, all on the objective side.
(i) Replace ε-prediction with **x-prediction** (predict the clean sprite), because the clean sprite lies on an
extremely low-dimensional manifold — 41 % of pixels are identical to a neighbour — while the noise does not.
(ii) Re-calibrate the noise schedule for this resolution by input scaling (a constant logSNR shift), instead of
inheriting a schedule tuned at 64–256 px. (iii) Drop the UNet for a plain transformer on pixels, since at
16×16 a UNet's two downsamples buy nothing.

**Closest prior work (verified).**
- Tianhong Li, Kaiming He. *Back to Basics: Let Denoising Generative Models Denoise.* arXiv:2511.13720
  (Nov 2025; rev. Jan 2026; a CVPR 2026 version appears on CVF — *flagged: I verified the arXiv page, which
  says "tech report", and saw a CVF CVPR-2026 PDF URL in search results but did not fetch it*). Argues
  precisely that the failure of pixel-space models is not the complexity of pixels but the standard practice
  of ε-prediction, and that plain transformers on raw pixels with x-prediction, no tokenizer, no pre-training
  and no extra loss are strong generative models.
- Ting Chen. *On the Importance of Noise Scheduling for Diffusion Models.* arXiv:2301.10972 (Jan 2023, tech
  report). Optimal noise scheduling depends on image size; scaling the input by a constant b (equivalently
  shifting logSNR by log b) is the right knob.

**Why it could beat 7.53 here.** The project has never varied the objective: every probe in the log is an
ε-prediction 100-step DDPM. Both of the above papers say, from opposite directions, that this exact choice is
the thing that goes wrong in pixel space, and Chen's result says the schedule is resolution-dependent, which
means the inherited schedule is almost certainly mis-set at 12–32 px with 41 % flat content. The manifold
argument bites unusually hard here: the ambient space at 16 px is 1024-D but the effective dimension of a
sprite is maybe 100–200.

**Cheapest falsifying probe.** Three 40 k-step runs (≈4 h each, or 20 k fine-tunes at ≈1.5 h each), identical
except for the objective: (a) v7h recipe as-is; (b) x0-prediction, same schedule; (c) x0-prediction, input
scaling b ∈ {0.5, 2} — pick one after a 2 k-step loss-curve peek. **Decision number:** bare matched FD. If (b)
or (c) does not beat (a) by ≥4 points, the objective is not the bottleneck and this whole line closes.

**Biggest reason it might fail.** None, in the sense that it will produce a number; the risk is that it
produces a *good* number and no paper. Novelty is zero (see §2). Also x0-prediction can under-perform at very
high noise levels, so the loss weighting must be re-tuned at the same time, which adds a nuisance dimension
to a "cheap" probe.

**Note on sequencing:** this is the one candidate that every other candidate inherits. A1 and A3 are already
x0-ish/discrete by construction; A2 and A5 are not. Running A4 first makes all later comparisons honest and
costs one day.

---

### A5 — Next-scale autoregression over the resolution ladder (VAR-style), tokens = pixels

**Mechanism.** Generate 12 → 16 → 20 → 24 → 32 as a sequence of full token maps, each conditioned on all
coarser maps, rather than denoising each independently. Total sequence over the ladder is ~2.4 k tokens —
trivial. The ladder is the model's own coarse-to-fine prior, and the 7.53 guidance trick becomes an explicit
conditional rather than a sampling-time hack.

**Closest prior work (verified).**
- Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang. *Visual Autoregressive Modeling: Scalable Image
  Generation via Next-Scale Prediction.* arXiv:2404.02905 (Apr 2024) — *flagged: the arXiv abstract page does
  not state the venue; NeurIPS 2024 by common knowledge, unverified from the page.*
- David Eigen. *Progressive Checkerboards for Autoregressive Multiscale Image Generation.* arXiv:2602.03811
  (Feb 2026, rev. Jul 2026). Multiscale AR with a balanced quadtree ordering; finds that a wide range of
  scale-up factors give similar results at fixed serial-step count — directly relevant, because it says the
  *choice of ladder* matters less than you would hope.

**Why it could beat 7.53 here.** The task has a real resolution ladder and the project's strongest measured
effect is cross-resolution. Explicit coarse→fine conditioning is the textbook way to turn that into training
signal.

**Cheapest falsifying probe.** You already ran most of the falsifier. `cycle 4` measured the 32→16 BOX
control: v7 84.07, tv 52.82 — coarse-derived-from-fine is *worse*. Before training anything, run the
complementary oracle: condition a 20 k-step fine-tune of v7h on the **ground-truth 12 px version** of the
target sprite and measure matched FD@16 (excluding oracle sources from training, per the `probe_paltok`
contamination lesson). **Decision number:** if oracle-conditioned FD is not below ~8, then even a perfect
coarse stage does not make the fine stage easy, and A5 is dead for the same reason `probe_sgen` was
(19.17 structure-domain FD against a 1.40 floor: the decomposition does not make the problem easier).

**Biggest reason it might fail.** It is a staged factorisation, and staged factorisation is 2-for-2 falsified
in this project (`probe_sgen` two-stage 61.66 with documented exposure bias; the BOX control above). Worse,
§0.1 says the coarse rungs of the ladder are *synthetic BOX downsamples of the same asset*, so the coarse
stage carries almost no independent information — it is a deterministic function of the target, which makes
the AR factorisation nearly degenerate at training time and badly exposure-biased at test time.

---

### A6 — Matryoshka-style joint multi-resolution denoising (**downgraded — likely already falsified**)

**Mechanism.** One nested model that denoises 12/16/20/24/32 versions of the same sprite jointly with shared
weights, so cross-resolution consistency is a training-time constraint rather than a sampling rule.

**Closest prior work (verified).** Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Josh Susskind, Navdeep Jaitly.
*Matryoshka Diffusion Models.* arXiv:2310.15111 (Oct 2023; "Accepted by ICLR2024", stated on the page).
NestedUNet, multiple resolutions denoised jointly.

**Why it might have worked here.** It internalises the mechanism behind the current best result.

**Why I am ranking it low anyway.** Two independent kills already on file. `probe_cg` trained a second label
set whose target is the 2×2 block average and it *hurt* monotonically in w (21.53 / 35.22 / 59.38), with the
recorded conclusion that an explicit low-pass branch is the wrong weak reference because it injects block
structure. Matryoshka's low-resolution branch is exactly that: in MDM the coarse input is a downsample of the
fine one, and per §0.1 the coarse rungs here are literally `Image.BOX` downsamples. There is no independent
low-resolution *asset* to couple to. **Cheapest falsifier before writing any code:** count how many
`data/oga_clean` entries are genuinely the same character drawn at two different native sizes. If that number
is under ~5 k pairs, A6 has no data to stand on and should be closed. **Decision number:** pair count ≥ 5 k
*and* a 20 k joint fine-tune beating v7h bare by ≥4; otherwise close.

---

### A7 — Adversarial / distribution-matching head at native resolution (**ceiling probe, zero novelty**)

**Mechanism.** Attach a small discriminator directly on the model's x0 prediction at native resolution and
add a relativistic adversarial term to the denoising loss. At 16×16 a discriminator is almost free.

**Closest prior work (verified).** Yiwen Huang, Aaron Gokaslan, Volodymyr Kuleshov, James Tompkin.
*The GAN is dead; long live the GAN! A Modern GAN Baseline.* arXiv:2501.05441 ("Accepted to NeurIPS 2024",
stated on the page). Regularised relativistic loss with local convergence guarantees; beats StyleGAN2 on
CIFAR and Stacked MNIST — i.e. at exactly the small-canvas regime in question.

**Why it could beat 7.53 here.** FD-DINOv2 at native resolution with one patch per pixel is essentially a
distribution-matching score on per-pixel colour and local contrast statistics. A discriminator optimises that
family of statistics directly, and mean regression — the diagnosed defect — is the one failure mode a
discriminator is guaranteed to punish. Of everything on this list, this has the highest probability of
*moving the number*.

**Cheapest falsifying probe.** 20 k-step fine-tune of v7h with a 4-layer patch discriminator on x0,
adversarial weight swept over {0.01, 0.05, 0.2} in one run each. **Decision number:** bare matched FD ≤ 12,
and — mandatory — the gain must survive re-scoring after post-hoc q16, or it is the `probe_selfq` outcome
again.

**Biggest reason it might fail.** Reviewers will say you optimised the metric: a discriminator on 16 px RGBA
and an FD on 16 px RGBA measure nearly the same thing. Use it as a **ceiling probe** — it tells you how much
of the 7.53 → 3.45 gap is reachable at all with this data and compute — and not as the paper's method.

---

## 2. Novelty collision, per candidate

| # | Nearest existing paper | Would an ICLR reviewer call it incremental? |
|---|---|---|
| **A1** | MD4 (2406.04329) for the objective; Nemotron MDM (2606.29814) for GCE + token editing; MaskGIT (2202.04200) for decoding | **Yes, if pitched as "masked diffusion for pixel art"** — all three components exist. The defensible framing is not the model but the *finding*: that raw-pixel masked diffusion at 12–32 px beats continuous diffusion when the vocabulary is per-channel rather than a palette, together with the falsification of the palette-based discrete family (v6f/v_ord) and a measured vocabulary-floor protocol that predicts in advance which discretisations can possibly work. That is an empirical-science paper, not an architecture paper. Note Nemotron is on VQ latents, so **raw-pixel MDM at native low resolution is genuinely unoccupied.** |
| **A2** | SDXL micro-conditioning (2307.01952) | **Yes, as a method.** Conditioning a diffusion model on the source resolution is SDXL's trick with the sign flipped. Publishable only as diagnosis: "low-resolution sprite corpora are unlabelled mixtures over a downscale factor; the mixture bias is what resolution-based guidance was accidentally repairing." Strong as §5 of a paper, weak as §3. |
| **A3** | MAR (2406.11838) for the backbone; pointer-generator (1704.04368) for the head; Infinity (2412.04431) as the other "escape the vocabulary" idea | **No — this is the one that is actually open.** I searched four ways for a copy/pointer head in image generation and found nothing. The collision risk is that a reviewer calls it "a pointer network, but for pixels", which is a one-line summary of the contribution; the defence has to be that exact colour reuse is a *domain-correct* inductive bias with a measurable, interpretable switch statistic that matches a measurable data statistic. |
| **A4** | JiT (2511.13720), Chen (2301.10972), NoiseShift (2510.02307) | **Yes, unambiguously.** Zero novelty. Do it anyway, as the honest baseline everything else is measured against. |
| **A5** | VAR (2404.02905), Infinity (2412.04431), Progressive Checkerboards (2602.03811) | **Yes.** Next-scale AR is a crowded 2024–2026 area and "VAR on sprites" is a workshop paper. Only interesting if the ladder were real assets, which §0.1 says it is not. |
| **A6** | Matryoshka (2310.15111) | **Yes, severely.** MDM already denoises multiple resolutions jointly with shared weights. The only difference would have been "the resolutions are distinct real artworks, not downsamples", and in this corpus they are downsamples. |
| **A7** | R3GAN (2501.05441), and the whole ADD / Diffusion-GAN line | **Yes.** Adding a discriminator to a diffusion model is 2022 technology. Ceiling probe only. |

---

## 3. Ideas I considered and rejected

- **Region/segment-set generation with differentiable rasterisation** (emit N coloured regions, rasterise) — searched; the differentiable-rasterisation literature is all meshes, CSG and Gaussian splats, nothing that fits a 256-pixel canvas. Also fails §0.1: flat-by-construction overshoots a target that is 92 % antialiased.
- **Quadtree / run-length token sequence** (flatness as literal encoding length) — same §0.1 objection, plus a variable-length sequence whose length correlates with content, which is a training-stability problem inside a 4 h budget.
- **Per-image palette in any form** (joint palette + index map, palette as pointer, ordered palette manifold) — the project killed three variants; §0.1 explains why the whole family is mis-targeted. Closed.
- **Full-pixel attention / pixels-as-tokens as the contribution** — already excluded by the project as an ImageGPT + PiT collision (Duy-Kien Nguyen, Mahmoud Assran, Unnat Jain, Martin R. Oswald, Cees G. M. Snoek, Xinlei Chen, *An Image is Worth More Than 16x16 Patches: Exploring Transformers on Individual Pixels*, arXiv:2406.09415, "In Proceeding of ICLR'2025", verified). Keep it as a *component* of A1/A3/A4, never as the claim.
- **Pure raster-scan per-pixel AR (ImageGPT-style)** — folded into A1 as the comparison arm rather than a candidate. Xinchen Yan, Chen Liang, Lijun Yu, Adams Wei Yu, Yifeng Lu, Quoc V. Le, *Rethinking Generative Image Pretraining: How Far Are We From Scaling Up Next-Pixel Prediction?*, arXiv:2511.08704 ("Accepted by ICML2026", verified) trains next-pixel Transformers at 32×32 up to 7e19 FLOPs and concludes the bottleneck is **compute**, forecasting five more years before pixel-by-pixel modelling is practical. A direct warning against betting a 4 h budget on raster AR; MD4's result that masked diffusion beats same-size AR at the pixel level points the same way.
- **Neural-field / arbitrary-scale decoder** (one latent sprite rendered at any R) — attractive story for a multi-resolution corpus, but continuous fields are smooth by construction, the opposite of the requirement, and there is no cheap way to make the field piecewise-constant without reinventing the killed palette family.
- **Exact-likelihood EBM / discrete MRF over the whole canvas** ("16×16 is small enough to model almost exactly") — the state space is 256^1024, not small; "small canvas" is an illusion once the vocabulary is realistic. Partition-function estimation at that size in 4 h is not happening.
- **Consistency / flow-matching / few-step distillation** — orthogonal to quality; the project is not compute-bound at sampling time.
- **Equivariant or symmetry-aware denoiser** (many sprites are mirror-symmetric or come in 4/8-directional sets) — plausible small win, but it is an augmentation-level effect with no mechanism against the diagnosed mean-regression defect.
- **Retrieval-augmented generation** (condition on nearest real sprites) — the corpus is 37 k images and the model already reproduces 8.4 % of it near-pixel-perfectly (recorded in `autonomy_state.md`); retrieval would make the contamination problem catastrophic and the FD meaningless.
- **DINOv2-feature auxiliary loss** — training on the metric. Would win, and would be indefensible.
- **Hand-crafted TV / flat-region priors** — already established as effective and unpublishable; keep as the baseline row (42.82 on the old protocol) and nothing more.
- **PixelFlow-style pixel-space cascade** (Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, Ping Luo, *PixelFlow: Pixel-Space Generative Models with Flow*, arXiv:2504.07963, verified) — the pixel-space part is already true here (this project never used a VAE); the cascade part is A5, which §0.1 undermines. Nothing left over.

---

## 4. Top recommendation

**Run A4 (one day) and A2 (one day) as the diagnostic pair, then bet on A1 as the architecture.**

The project has spent seven cycles varying the *guidance* and the *prior* while holding the objective and the
output parameterisation fixed, and the two facts in §0 say the remaining error is plausibly (a) a mixture bias
the model was never given the variable to resolve, and (b) an ε-prediction objective that two independent
2023–2026 results identify as the specific thing that breaks pixel-space models. Both are cheap to test and
both change what "beating 7.53" even means, because if A4 alone moves the bare number from 21.98 to 12, the
composed-guidance baseline moves too and the bar moves with it. A1 is the architecture bet because it is the
only candidate that attacks *structure* rather than colour — the per-pixel independence of the DDPM reverse
step is a real, unaddressed defect at 256 pixels, and the chain rule removes it exactly — and because the
previous kill of that family is confounded by a 32-entry global palette in a corpus whose median sprite uses
34 colours, a confound the ten-minute vocabulary-floor check will expose before a single GPU-hour is spent.
A3 is the more original paper and should be built on A1's infrastructure if A1's decoding works at all, since
it is the same backbone with a different head.

**Honest probability that A1, with guidance, clears 7.53: ~30 %.** Decomposed: ~70 % that the vocabulary-floor
check passes and the recipe survives contact with RGBA and text conditioning; ~55 % that masked diffusion
reaches bare FD ≤ 12, given MD4's pixel-level result and the identified confound in v6f; then ~55 % that the
guided version of that beats a guided baseline which will itself have improved if A4 works. Dominant risks:
per-channel factorisation producing off-manifold colours that FD punishes at one-patch-per-pixel, and the
possibility that the residual 7.53 → 3.45 gap is coverage/diversity that no output parameterisation fixes
(the existing `fd_decomp` finding that the gap is a coverage term is evidence *against* A1 and should be
re-checked before committing).

For calibration, if the goal ever shifts from an ICLR architecture to a number on a leaderboard, the ranking
inverts: **A7 has ~55 % of clearing 7.53 and ~10 % of being publishable; A2 has ~45 % of clearing it and
~20 % of being publishable as a method.**

---

## Appendix — citation verification status

Verified by fetching `arxiv.org/abs/<id>` on 2026-09-09 (title, authors, dates, abstract read in full):
2511.13720, 2406.04329, 2406.11838, 2510.02307, 2602.03811, 2301.10972, 2404.02905, 2310.15111, 2202.04200,
2308.04052, 2406.09415, 1704.04368, 2412.04431, 2410.06236, 2511.08704, 1701.05517, 2606.29814, 2504.07963,
2307.01952, 2501.05441, 2510.22510.

Partial / flagged:
- **2307.01952 (SDXL)** — paper verified; the *original-image-size micro-conditioning* detail is in the body,
  not the abstract, so that specific claim is from memory of the paper, not from the fetched page.
- **1704.04368 (pointer-generator)** — page gives no venue; ACL 2017 is from memory.
- **2404.02905 (VAR)** — page gives no venue; NeurIPS 2024 (best paper) is from memory.
- **2511.13720 (JiT)** — arXiv page says "tech report"; a CVPR 2026 CVF PDF URL appeared in search results but
  was not fetched, so treat the venue as unconfirmed.
- Cited for context only, not fetched: **Muse (2301.00704)**, **MDLM (2406.07524)**. Do not cite either
  without fetching first.

Also verified but used only as background/collision references: *The Five-Dollar Model: Generating Game Maps
and Sprites from Sentence Embeddings* (Timothy Merino, Roman Negri, Dipika Rajesh, M Charity, Julian
Togelius, arXiv:2308.04052, "To be published in AIIDE 2023") and *SD-πXL: Generating Low-Resolution Quantized
Imagery via Score Distillation* (Alexandre Binninger, Olga Sorkine-Hornung, arXiv:2410.06236, SIGGRAPH Asia
2024) — the latter's Gumbel-softmax convex-sum-over-palette generator is the closest published thing to any
"hard palette head", and per §0.1 it is targeting a different objective (user-specified palette) than FD
against a mostly-downscaled reference.

**Note on `PixDiff-PIG`:** a ResearchGate entry titled *PixDiff-PIG: Palette-Informed Diffusion for Pixel Art
Generation* still surfaces in search, but the 2026-09-09 (04:45) citation audit already recorded it as
unfindable and deleted it from the draft. I could not find an arXiv or venue record for it either. Keep it
deleted.
