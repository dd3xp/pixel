# Turning cross-resolution self-guidance into an architecture: candidates, novelty audit, recommendation

Date: 2026-09-12. Scope: 12 / 16 / 20 / 24 px (32 px appendix at most). Base model: v7h (72.5 M
`UNet2DConditionModel`, eps-prediction, 7-bucket class embedding added to the timestep embedding).
Every citation below was fetched on 2026-09-12. Where only the abstract could be read, the entry says so
(Section 9). No other repo file was changed.

---

## 0. Summary

**Success bars (the coordinator's revised version, used throughout):**

- **PASS**: new architecture + plain CFG beats UNet + plain CFG at 16 px (FD-DINOv2 < 12.49; I also require
  it to hold at matched NFE).
- **STRONG**: new architecture + cross-resolution guidance < 7.53.
- **INTERNALISER**: a design that folds the guidance in counts as a contribution if it gets to about 7.53 in
  **one** forward pass with no early snapshot stored at inference.

**Two findings decide most of this report:**

1. **The MSE fixed-point argument (Section 1).** Train a module on the standard denoising loss over
   forward-noised data and its population optimum is still `E[eps | x_t, c, b]`. The lower-bucket belief is a
   deterministic function of `(x_t, c)`, so it adds no information at the Bayes optimum. There are then only
   two ways a trained module can "learn the guidance":
   - (i) The finite model's own error can be predicted from the difference `e_b - e_low`. This is measurable
     at zero training cost.
   - (ii) The loss is rewritten so that its fixed point *is* the guided prediction. That is the MG / GFT /
     SGG / distillation family, and a reviewer will file the result there.

   Candidate 3 lands in family (ii) exactly. I prove this in Section 4.
2. **Nearly everything obvious is already published** (all verified):
   - A one-pass internal weak branch: Internal Guidance (IG, Zhou et al., arXiv 2512.24176) and SGG-BR
     (Yuan et al., arXiv 2603.20584).
   - Moving a weak-model guidance direction into the training target: MG (Tang et al., arXiv 2502.12154),
     SGG (training migration) and GFT (Chen et al., ICML 2025).
   - Learned guidance weights: Galashov et al. (arXiv 2510.00815) and Pokle et al. (arXiv 2608.14038).
   - Training on self-rolled states: SOAR (arXiv 2604.12617).
   - A same-network degraded *condition* as the negative (this hits the sampling paper, not the
     architecture): Han et al., CVPR 2026 (arXiv 2603.10780). It is not cited in the current draft and should
     be.

**Verdicts on the three candidates:**

| # | Candidate (as briefed) | Novelty | Most plausible bar | Inference NFE |
|---|---|---|---|---|
| 1 | Built-in dual-resolution branch + learned combination, one pass | **Incremental.** A shared-trunk dual tail is IG with a resolution-conditioned head. A learned combiner is either Meng-style distillation (guided target) or collapses to `w ~ 1` (MSE target). In GFT form (1b) it is GFT with a new reference. | 1b could meet the INTERNALISER bar (~35-40 %) | 1b: 1.0; 1a: ~1.23-1.5 |
| 2 | Cross-resolution consistency module | **Depends on the form.** (a) Features from an actually downsampled input: collides with Matryoshka, and our blur-input result (FD 223) argues against it. (b) Features from a *counterfactual lower label on the same `x_t`*, fused into the target pass and trained on data targets only: **no prior art found**. | PASS ~40 %, STRONG ~20-25 % | 2 (the context pass doubles as the guidance reference) |
| 3 | Resolution-contrastive head / loss | **Fails novelty.** Its fixed point is the MG target with `w = 1/(1-lambda)` (proof in Section 4). Contrastive/InfoNCE variants are CCA. | PASS very likely (~70 %) | 1 |

**Candidates I added:**

- **4. On-policy counterfactual correction.** Train on the model's own one-step rollouts, with the
  counterfactual belief as an input. Standalone it collides with SOAR, so I fold it into the recommendation as
  a training-recipe switch.
- **5. One-NFE class-embedding extrapolation.** A zero-training diagnostic.
- **6. An 8 px ladder rung**, so every reference-based method also works at 12 px.

**Recommendation: CRSC, Counterfactual-Resolution Self-Conditioning (candidate 2, form b).**

- **Architecture.** v7h gains two things, both zero-initialised:
  - an input channel group carrying its own lower-bucket `x0` belief for the *same* `x_t`;
  - two decoder adapters that read the lower-bucket pass's decoder features.
- **Training.** Data targets only. No guided output is ever a regression target, which is what separates it
  from distillation, MG and GFT.
- **Inference.** 2 NFE: pass 1 = lower label, pass 2 = target label with the context. The pass-1 output is
  also the guidance reference, so "CRSC + cross-resolution guidance" costs exactly what today's label guidance
  costs, with no snapshot.
- **Gate.** Before any training, run the zero-training diagnostics T0-T3 (Section 5; about 1-1.5 GPU-hours).
  They say whether the error signal CRSC needs actually exists.
- **Fallback.** If it does not, switch to candidate 1b: GFT-form internaliser, 1 NFE. It is incremental on
  novelty but is the right design for the INTERNALISER bar, and the paper needs it as a baseline anyway.

---

## 1. The fact that decides which designs can work: what a data-trained module can learn

Notation:

- `e_b = eps_theta(x_t, t, c, b)` is the strong belief.
- `r = eps_theta(x_t, t, c, b_low)` is the lower-bucket belief. For the composed reference, `r` is instead the
  10 k snapshot at `b_low`.
- The guided prediction is `g_w = r + w (e_b - r)`.
- `mu = E[eps | x_t, c, b]` is the Bayes denoiser.

**(a) Plain denoising loss on forward-noised data.**

- For any architecture, the population minimiser of `E||f(x_t, c, b, extra) - eps||^2` is
  `E[eps | x_t, c, b, extra]`.
- If `extra` is computed from `(x_t, c)`, as the lower-bucket belief is, this equals `mu`.
- Guidance with `w != 1` is **not** `mu`. It is useful because the *finite* network is not `mu`: Karras et
  al.'s "compatible errors" argument.
- So a data-trained module can learn to extrapolate away from `r` only as far as the finite model's residual
  `mu - e_b` is predictable from `d = e_b - r`. The linear version is directly measurable:

  `w*(t) = E[ d . (eps - r) ] / E[ d . d ]`, with every expectation over held-out `(x0, eps)` at fixed `t`.

  This estimator is unbiased for `E[d . (mu - r)] / E||d||^2`, because `eps - mu` has zero mean given `x_t`.
- `w*(t) ~ 1` means the error is not linearly visible in the lower-bucket belief. A module trained with MSE
  will then not internalise the guidance, whatever its architecture.
- `w*(t) >> 1` means the network's own error is partly readable from its counterfactual belief. A module can
  then legitimately learn this, and learn it better than a single global `w`.

**(b) The model's own trajectory states.** The observed deficit (colour mixing, the FD mean term) may build up
along the sampler rather than sit in `mu`'s approximation error.

- On states `x~_t` reached by rolling the deployed sampler forward from a noised real `x0`, the regression
  target `x0` is still known. `w*` in `x0`-space is measured the same way.
- If `w*` is well above 1 here but not in (a), then the guidance is acting as a drift corrector. Training on
  rolled states (the SOAR recipe) is then the non-distillation route.

**(c) Anything else internalises the guidance by construction.** Examples: distillation targets, MG's
modified target, GFT's reparameterisation, repulsion losses. Such a model outputs `g_w` at its optimum. That is
legitimate engineering and is the correct design for the INTERNALISER bar, but it belongs to a named family.

This split is the novelty test used below. **A design is "not distillation" exactly when no guided
combination ever appears as its regression target or as its loss's fixed point.**

---

## 2. Candidate 1: built-in dual-resolution branch with learned combination (one forward pass)

### 2.1 Specification (three concrete readings)

**1a. Shared trunk, two label-conditioned tails (IG-style).**

- The encoder, mid block and `up_blocks[0..k-1]` run once with the target label.
- The last `3-k` up blocks and `conv_out` run twice, batched: once with `emb_b` and once with `emb_blow`.
- Output: `e = tail_low + w (tail_b - tail_low)`, with a fixed `w`.
- In diffusers this needs a custom forward: the same code path as `UNet2DConditionModel.forward`, but with a
  different `emb` from the split point on. The class embedding is simply added to the time embedding, as
  verified in the diffusers source: `emb = emb + class_emb`, `resnet_time_scale_shift="default"`.
- Training: none needed to test it (T3, Section 5). IG-style auxiliary supervision could be added afterwards.

**1b. GFT-form internaliser (exactly 1 NFE).**

- Add a scalar `beta` input: a Fourier embedding and MLP, zero-initialised, added to the time embedding, as in
  GFT Algorithm 1.
- Parameterise the conditional prediction as
  `eps_pred(x_t,c,b) = beta * s_theta(x_t,c,b,beta) + (1-beta) * sg[ r(x_t,c,b_low) ]`.
  - `r` is either the online network in plain mode (`beta = 1`) at the lower label, or the frozen 10 k EMA
    snapshot at the lower label (composed version; the snapshot is used **only in training**).
- Loss: `||eps_pred - eps||^2`, with `beta ~ U(0.4, 1)` and `beta = 1` for 25 % of samples to keep the plain
  model alive.
- Inference: `s_theta(x_t, c, b, beta = 1/w)`, 1 NFE. At the optimum
  `s = r + (1/beta)(mu - r)`, which is the guided prediction.
  - This follows GFT eq. 6 and eq. 9, with the unconditional term replaced by the lower-bucket reference.

**1c. Learned per-pixel combiner.**

- A small conv head (about 0.3 M parameters) reads `[h_b, h_low]` from the last decoder stage and outputs a
  non-negative map `W(x, t)` of shape `(1, H, W)`.
- Output: `e = r + W * (e_b - r)`.
- Training options, each discussed in 2.2:
  - (i) MSE on data;
  - (ii) regression to a 2-NFE guided teacher;
  - (iii) an adversarial / distribution-matching loss.

### 2.2 Closest work and verdict

- **1a vs Internal Guidance.** IG (Zhou et al., arXiv 2512.24176) gets a weak prediction `D_i` from an
  intermediate layer in the same forward pass, supervises it with the same `x0` target, and samples with
  `D_w = D_i + w (D_f - D_i)`. SGG-BR (Yuan et al., arXiv 2603.20584) uses an auxiliary branch as the
  condition-agnostic weak signal. SSG (Fu et al., arXiv 2607.29122) attaches a light intermediate head to a
  frozen pixel-space model.
  - 1a is "IG, but the weak head is resolution-conditioned instead of shallow".
  - **Verdict: incremental.** Its one distinct element is the counterfactual-condition tail. A reviewer will
    call it an IG variant.
- **1b vs GFT / MG / SGG.** GFT (Chen et al., ICML 2025, arXiv 2501.15420) is exactly this
  parameterisation, with a stop-gradient on the reference and the true `eps` target, and it explicitly
  positions itself as not distillation. MG (Tang et al., arXiv 2502.12154) and SGG's training migration reach
  the same fixed point through a modified target.
  - **Verdict: incremental (GFT with a new reference).** It is not guidance *distillation* in Meng's sense:
    there is no teacher and no guided target, only a reparameterisation trained on data targets. A reviewer
    will still call it "GFT/MG with a resolution reference". Its value is cost (INTERNALISER bar), not novelty.
- **1c.**
  - Trained against a guided teacher, it is guidance distillation (Meng et al., CVPR 2023, arXiv 2210.03142;
    plug-and-play variant with a light guide network: Hsiao et al., CVPR 2024, arXiv 2406.01954).
  - Trained with MSE, Section 1 shows it converges to `w*(t)` from T0, which is probably close to 1.
  - Trained adversarially, it is the per-pixel extension of Pokle et al. (arXiv 2608.14038: a scalar
    guidance-scale MLP on a frozen base, discriminator on noisy marginals) and Galashov et al.
    (arXiv 2510.00815: `w(c, s, t)` by distribution matching). Training-free spatial weighting already exists
    too: S-CFG (Shen et al., CVPR 2024, arXiv 2404.05384) and SAMG (Li et al., arXiv 2604.26503).
  - **Verdict: incremental in every training mode.**

**What would make candidate 1 not distillation:** the combination is never fitted to a guided output. That
holds for 1a and 1b, but both then fall into IG and GFT respectively.

### 2.3 Why it could meet a bar

- **1b** is the natural INTERNALISER.
  - Label-only guidance gives 8.52 at 2 NFE. The composed rule gives 7.53 at 2 NFE plus a stored snapshot.
  - 1b trained against `sg[snapshot(b_low)]` could reproduce the composed rule at 1 NFE, with the snapshot
    needed only during training.
  - GFT reports parity with CFG, not superiority, so expect roughly 7.5-8.5.
  - Estimated chance of ≤ 7.8 at 16 px: 35-40 %.
- **1a** can be at most as good as the full label switch (8.52), since it is a restricted version of it.
  Realistically 9-11 at about 1.23-1.5 NFE. It cannot meet the INTERNALISER bar and is not an architecture
  for PASS.
- **1c** can only beat the global-`w` rule if the per-pixel adaptivity carries real signal. The evidence
  says the useful direction is low-frequency, radial and in the colour channels; APG, FDG and channel
  decoupling all failed. Expect less than 0.5 FD from spatial adaptivity.

### 2.4 Cost

These are analytic MAC estimates for the v7h configuration at 16 px, to be confirmed with a profiler:

| Part | GMAC | Share |
|---|---|---|
| Encoder + mid | ~1.05 | ~34 % |
| `up_blocks[0]` | ~0.50 | ~16 % |
| `up_blocks[1]` | ~0.80 | ~26 % |
| `up_blocks[2]` | ~0.71 | ~23 % |
| Total | ~3.05 | 100 % |

- **1a:** 1.23 NFE (split before `up_blocks[2]`), 1.49 NFE (before `up_blocks[1]`), 1.66 NFE (before the
  decoder).
- **1b:** training is about +33 % over plain fine-tuning (one extra no-grad reference forward), so 20 k steps
  take about 4.7 h on the contended A100. For reference, probe_cg's 20 k-step fine-tune took 3.5 h. Inference
  is 1.0 NFE.
- **1c:** tiny module. The cost is in the objective: adversarial training needs rollouts.

### 2.5 Cheapest falsifier and decision number

- **1a: T3, zero training.** Sample with the decoder-only label switch as the reference, `w = 2`, 16 px,
  3000 matched prompts.
  - Retained gain = `(12.49 - FD_partial) / (12.49 - 8.52)`.
  - **Go** if retained gain ≥ 0.75 for the `up_blocks[2]`-only split. **Dead** if < 0.5 for the
    `up_blocks[1:]` split.
- **1b:** a 10 k-step fine-tune (about 2.5 h), then 1-NFE FD at `beta = 1/1.5` and `beta = 1/2`.
  - **Keep as the INTERNALISER row** if ≤ 8.5 (label guidance matched at half the cost).
  - **Claim the bar** if ≤ 7.8 over 3 seeds.
- **1c:** T0. If `w*(t) < 1.1` at all `t`, an MSE-trained combiner has nothing to learn.

### 2.6 Biggest reason it might fail

- **1a:** the label's effect is spread over every ResBlock bias (the class embedding enters every resnet
  through `time_emb_proj`), so a tail-only switch may carry little of it.
- **1b:** the online reference is a moving target, and the GFT extrapolation is applied at every `t`.
  Evidence: the snapshot only helps at `t/T ≤ 0.5`, and CFG is over-guided in this model. A 1-NFE model trained
  at one `beta` may over-sharpen at high noise. `beta` must be sampled, not fixed.
- **1c:** the MSE target has no reason to reward `w > 1` (Section 1).

---

## 3. Candidate 2: cross-resolution consistency module

"Features of a lower resolution of the same sample" can mean two different things, and they get opposite
verdicts.

### 3.1 Form (a): features from an actually downsampled input (Matryoshka-like)

**Spec.**

- Run a second stream on the actual BOX-downsampled `D(x_t)` at 12 px, label 12. At 16 px this costs about
  56 % of a 16 px pass.
- Inject its features into the target stream at matching decoder stages through cross-attention or upsampled
  addition.
- Or go further: jointly denoise a 12 px copy with its own noise.

**Closest work.** Matryoshka Diffusion (Gu et al., arXiv 2310.15111; published at ICLR 2024; the brief's
"2023" is the arXiv year) jointly denoises nested resolutions with a NestedUNet whose small-scale features
and parameters are nested in the large-scale ones.

**Verdict: collides.** The only difference would be the ladder range.

**It also runs against the evidence:**

- Low-resolution information is used *positively* (coarse to fine) in Matryoshka, whereas the finding says the
  lower belief is useful as a *negative* reference.
- Degrading the *input* rather than the label destroyed comparability: a 2x2 box-blurred `x_t` as reference
  gave FD 223.07; a 1 px shift gave 18.95.
- BOX-downsampling `x_t` also changes the noise variance per pixel, which is a mismatch the network was never
  trained on.

**Falsifier (zero training).** A reference built from `U(eps_12(D(x_t)))` at `w = 2`. My prediction is that it
behaves like the blur reference (> 20).

**Not recommended.**

### 3.2 Form (b): counterfactual lower label on the same `x_t`, fused into the target pass (CRSC)

**Spec (details in Section 7).**

- Pass 1 is the plain network at `b_low` on the same `x_t`, same caption, run without gradient in training.
  It yields `x0_low` and the decoder features `h_low` at `up_blocks[1]` and `up_blocks[2]`.
- Pass 2 is the network at `b`. It receives:
  - `x0_low` as 4 extra input channels, pixel-aligned by construction (same grid, same `x_t`), through
    zero-initialised `conv_in` weights;
  - zero-initialised adapters `h <- h + Z([h; h_low; h - h_low])` after `up_blocks[1]` and `up_blocks[2]`.
- Loss: plain `||e - eps||^2` against the true `eps`. Optionally the one-step rolled-state variant
  (Section 6.1).

**Closest work and how it differs.**

| Work | What it does | How CRSC differs |
|---|---|---|
| Self-conditioning (Chen, Zhang, Hinton, ICLR 2023, arXiv 2208.04202) | Feeds the model's own previous `x0` estimate back as input; training uses a stop-gradient first pass on the same `x_t` with 50 % zeroing | CRSC's context comes from a *different condition* (the lower resolution) at the *current* step. It is a counterfactual belief, not a previous-step estimate. |
| Matryoshka | Uses a real low-resolution image | No real low-resolution image is involved |
| IG / SSG / SGG-BR | Depth-weakened heads used only at sampling | The counterfactual belief is an *input* to the strong pass |
| NAG (Chen et al., arXiv 2505.21179) | Training-free extrapolation in attention-feature space | CRSC is trained, and its output is not constrained to be an extrapolation |
| MG / GFT / distillation | Guided prediction is a target or a fixed point | Nothing guided is ever a target |

- I found no paper that trains a diffusion network to take its own belief under a counterfactual condition
  of the same noisy input as context. That is three targeted searches, not proof of absence.
- **Verdict: novel as a module**, provided the ablation "lower-label context vs same-label context" comes out
  in its favour. If same-label context does as well, the method reduces to a two-pass self-refinement
  (self-conditioning on the same step) and the counterfactual claim goes.
- **What makes it not distillation:** the guided combination never appears as a target or fixed point. The
  lower belief is information the network may use in any way, including ignoring it.

**Why it could meet the bars.** The comparisons are NFE-matched.

- **PASS:** CRSC alone (pass 2 output, `w = 1`, 2 NFE, no CFG) vs UNet + best CFG (2 NFE, 12.49). The
  architecture has to buy about 24 %, starting from 16.39, which is v7h without guidance.
  - This is plausible if T0 or T1 shows `w* ≥ 1.3`: the network can then read its own error from the
    counterfactual and apply it adaptively.
  - If `w* ~ 1`, only the gain from extra computation remains. Self-conditioning shows this is not zero, but
    four points is a stretch.
- **STRONG:** CRSC + label guidance, using pass 1 as the reference so the cost stays at 2 NFE with no
  snapshot: `e = e_low + w (e_crsc - e_low)`. This can go below 7.53 if CRSC lowers the base error without
  absorbing the lower belief. The composed rule's covariance gain comes from the snapshot; CRSC would need to
  supply that through a better base.
  - Estimates: PASS ~40 % overall (~65 % if T0 or T1 is positive, ~20 % if both are negative); STRONG ~20-25 %.

**Cost.**

- Training with context on 50 % of samples (one extra no-grad forward for those): about 1.17x a plain
  fine-tune, so 20 k steps take about 4.1 h.
- With the rolled-state recipe on 25 % of samples (+2 no-grad forwards for those): about 1.35x, so 20 k
  steps take about 4.7 h.
- Inference: 2 NFE. 3 NFE with CFG (uncond branch without context); 3 NFE with the composed snapshot
  reference.

**Cheapest falsifier.**

- T0 and T1 (zero training, under 1 GPU-hour).
- Then a 5 k-step smoke run (about 1 h) with two readouts:
  - (i) held-out eps-MSE with context vs without context on the same weights. Continue if the relative
    reduction is ≥ 1.0 % for `t/T ≤ 0.5`. Kill if < 0.3 %.
  - (ii) the sign of the learned use: regress `e_ctx - e_noctx` onto `e_noctx - e_low`. Continue if the
    slope is > 0.1 (repulsive, as the finding predicts). Kill if < 0 (absorption).

**Biggest reason it might fail: absorption.**

- Nothing in the loss says the counterfactual must be used repulsively.
- The easiest way to reduce MSE with a structure-aligned, low-contrast copy of one's own prediction may be to
  lean *towards* it (Matryoshka-style coarse-to-fine use).
- That shrinks `e_b - e_low`, so the guidance on top gains less, and the total can come out flat.

---

## 4. Candidate 3: resolution-contrastive head / auxiliary objective

### 4.1 Specification

- Add a repulsion term against a stop-gradient lower-bucket reference:
  `L = ||e_b - eps||^2 - lambda * ||e_b - sg(r)||^2`, with `0 < lambda < 1`.
  - `r` is the EMA network at `b_low`, as in MG, or the 10 k snapshot at `b_low` (composed version).
- Variants: a margin/triplet version, or InfoNCE over buckets, where the prediction should be
  "classifiable" as bucket `b` rather than `b_low`.
- Inference: 1 NFE.

### 4.2 The fixed point settles the novelty question

At fixed `x_t`, minimising `E||e - eps||^2 - lambda ||e - r||^2` over `e`:

- The stationarity condition is `2(e - mu) - 2 lambda (e - r) = 0`.
- So `e* = r + (mu - r) / (1 - lambda)`, i.e. **the guided prediction with `w = 1/(1-lambda)`**.
  `lambda = 0.5` gives `w = 2`; `lambda = 1/3` gives `w = 1.5`.
- The curvature is `2(1 - lambda)`, so for `lambda ≥ 1` the loss is unbounded below and training collapses.
- MG's target `eps + w_MG * sg(e_c - e_u)`, built from the EMA network, has the fixed point
  `mu + w_MG (e_c - e_u) ~ e_u + (1 + w_MG)(mu - e_u)`. That is the same family, with `1 + w_MG = 1/(1-lambda)`.
- SGG's training migration (`u + w * sg[g]`, where `g` can come from an auxiliary branch) is the same again.
- The InfoNCE-over-buckets variant pushes along `grad log p(b | x_t) ∝ -(e_b - e_low)`: it is CCA (Chen et
  al., arXiv 2410.09347), contrasting conditions instead of conditional vs unconditional.

**Verdict: guidance self-distillation (MG / SGG / CCA family).** The only new element is the choice of
reference. Under the brief's rule, this is a failure on novelty whatever its FD.

### 4.3 Why it could meet a bar, cost, falsifier, failure mode

- **Bar:** PASS is very likely, about 70 %. It bakes label guidance (8.52 at 2 NFE) into 1 NFE, and CFG on
  top is optional. STRONG is unlikely without the snapshot. As an INTERNALISER it is equivalent to 1b in
  expectation.
- **Cost:** +33 % training (the reference forward), so 20 k steps take about 4.7 h. 1 NFE at inference.
- **Falsifier:** a 10 k-step fine-tune, then 1-NFE FD. Decision: ≤ 12.49 means PASS; ≤ 8.5 means the label
  rule is matched at half the cost.
- **Failure mode:** the lower bucket is repelled from itself while also being trained as a plain belief. The
  reference must come from the EMA or the snapshot, or it drifts and the effective `w` runs away.
- **Use in the paper:** an ablation row labelled "MG-form internalisation".

---

## 5. Zero-training diagnostics: run these first (total about 1-1.5 GPU-hours)

All at 16 px on v7h EMA, with the held-out protocol (the 5,928 excluded sprites and their captions), then
repeated at 20 and 24 px.

**T0: MSE-optimal extrapolation weight on forward-noised held-out data.**

- **Procedure:**
  - Take 10 timesteps from the 100-step schedule, evenly spaced in `t/T`.
  - For each `(x0, c)`: draw `eps`, form `x_t`, and compute `e_16`, `e_12`, and optionally
    `e_snap12 = snapshot(x_t, c, 12)`.
  - Report per `t`, each with a bootstrap CI:
    - `w*(t)` from Section 1 for the label, snapshot and composed references;
    - the relative loss change `L(w)/L(1)` at `w = 1.5` and `w = 2`.
  - Repeat on 5,928 *training* sprites, to separate generalisation effects from approximation effects.
- **Cost:** about 6 k × 10 × 3 forwards at batch 500, a few minutes.
- **Decision:**
  - If `w*(t) ≥ 1.3` over at least 30 % of the steps (for any reference), MSE-trained modules have a signal:
    CRSC is trained on forward-noised states.
  - If `w* < 1.1` everywhere, the guidance is not a posterior-error correction on the data distribution.

**T1: the same on self-rolled states.**

- **Procedure:**
  - Noise `x0` to `t' = t + k` sampler steps.
  - Run the deployed sampler for `k` steps, both bare and label-guided, to reach `x~_t`.
  - Compute `w*` in `x0`-space against the known `x0`, for `k ∈ {1, 5, 20}`.
- **Decision:** if `w* ≥ 1.3` for `k ≥ 5` while T0 is below 1.1, the guidance is a drift corrector: CRSC must
  use the rolled-state recipe, and "SOAR alone" becomes the key ablation.
  - If T0 and T1 are **both** below 1.1: no data-trained module will internalise the guidance. Drop CRSC as
    the headline; run 1b as the INTERNALISER; keep the architecture section honest.

**T2: one-NFE class-embedding extrapolation (candidate 5).**

- **Procedure:** query v7h once per step with `class_emb = E[16] + (w - 1)(E[16] - E[12])`, for
  `w ∈ {1.5, 2}`. Sample 3000, compute FD.
  - The label enters only through `emb = temb + class_emb`, so this is a one-line change.
- **Decision:** FD ≤ 10 at 1 NFE would be a surprising, free result and a strong baseline that every
  internaliser must beat. FD > 16.39 (worse than no guidance) closes it.
- **Novelty:** my one search found no paper on output-equivalent extrapolation in condition-embedding space.
  Treat this as unverified.

**T3: decoder-only label switch (feasibility of 1a).** Defined in 2.5.

---

## 6. Added candidates in brief

### 6.1 Candidate 4: on-policy counterfactual correction

- **What it is:** CRSC trained additionally on one-step self-rolled states, where the target is the original
  `x0`, re-expressed as `eps~ = (x~_t - sqrt(abar_t) x0) / sqrt(1 - abar_t)`.
- **Closest work:**
  - SOAR (Qin et al., arXiv 2604.12617): a single stop-gradient rollout, re-noising, and supervision back
    towards the clean target.
  - Exposure-bias training: input perturbation (Ning et al., ICML 2023, arXiv 2301.11706) and epsilon
    scaling (Ning et al., ICLR 2024, arXiv 2308.15321).
  - Epsilon scaling is a global output rescale. It is the exposure-bias analogue of our failed `shrink:f`
    control, and worth citing next to it.
  - Discriminator guidance (Kim et al., ICML 2023, arXiv 2211.17091) learns a score correction from
    real-vs-generated samples.
- **Verdict:** standalone it is SOAR plus an extra input. It is only worth keeping as CRSC's training-recipe
  switch, triggered by T1, with **SOAR-alone** (rolled states, no context) as the required control.

### 6.2 Candidate 6: an 8 px ladder rung for 12 px

- **What it is:** grow the class embedding from 7 to 8 rows, initialise the 8 px row from the 12 px row, and
  feed every sprite of at least 10 px to an 8 px bucket.
- **Why:** every reference-based method, the current sampling rule included, is undefined at 12 px because no
  lower bucket exists. This row fixes that.
- **Status:** not a contribution, an enabling change.
- **Risk:** as with the 12 px reference at 20 px, a far rung drifts structurally (opacity). The fallback is a
  10 px rung.

---

## 7. Recommendation: CRSC, full implementation plan

### 7.1 Files

Existing patterns: `train_coarse.py` imports from `train_v7.py`; `sample_e.py` has a `guide_mode` switch;
server jobs run under tmux via `supervise.sh`.

| File | Change |
|---|---|
| `src/v6/diag_mse_w.py` (new) | T0 / T1 / T3 diagnostics. Loads v7h EMA and the 10 k snapshot, held-out rows from the exclude list, outputs `runs_out/diag_mse_w_<size>.json`. |
| `src/v6/crsc_unet.py` (new) | The model wrapper (7.2). |
| `src/v6/train_crsc.py` (new) | Training loop (7.3). Reuses `NativeSprites`, `BucketSampler`, `embed`, `make_grid` and `BATCH` from `train_v7.py`. |
| `src/v6/sample_e.py` (edit) | New `--guide_mode crsc[:wref]`, `--ctx_label`, `--emb_extrap` (T2). Loading detects `conv_in.weight.shape[1] == 8`. |
| `baseline/run_crsc.sh`, `baseline/eval_crsc.sh`, `baseline/run_diag_mse_w.sh` (new) | tmux + `supervise.sh` wrappers. Final lines CRSC_DONE / DIAG_DONE, logs in `logs/`. |
| `src/v6/train_gft_res.py` (new, fallback / baseline) | Candidate 1b. |
| `src/v6/train_mg_res.py` (new, baseline) | Candidate 3 / MG-form. |
| `src/v6/train_distill_res.py` (new, baseline) | Meng-style: 1-NFE student regresses the 2-NFE label-guided or composed teacher output. |

### 7.2 Module structure (`crsc_unet.py`, in prose)

**Model.** `CRSCUNet(nn.Module)` holds:

- a `UNet2DConditionModel` with `in_channels=8` and `num_class_embeds=8` (the 7 v7h buckets plus 8 px);
- two adapters, `CRF1` (256 channels, after `up_blocks[1]`, 8×8 at 16 px) and `CRF2` (128 channels, after
  `up_blocks[2]`, 16×16);
- a time-embedding projection per adapter.

**`from_v7h(state_dict)`:**

- Copy every tensor.
- `conv_in.weight`, new shape `[128, 8, 3, 3]`: the first 4 input channels are v7h's, the last 4 are zero.
- `class_embedding.weight` goes from 7 to 8 rows; the new 8 px row is a copy of the 12 px row. Keep the
  existing index order and append 8 px as index 7, so all v7h label indices stay valid.
  - Then `NativeSprites`' native-bucket lookup (`next(i for i, bb in enumerate(BUCKETS) if bb >= s)`) must
    skip index 7, and the augmentation list `LOW` gains index 7 for sprites with `s >= 10`.
- At initialisation the model is exactly v7h whenever the context is zero.

**Adapter `CRF(C)`:**

- `u = SiLU(GroupNorm(Conv1x1(3C -> C)([h; h_low; h - h_low])))`
- `u = u * (1 + Lin(temb))`, a per-timestep channel gate
- `h <- h + ZeroConv1x1(C -> C)(u)`

Parameters: about 0.35 M in total. Zero-initialising the last conv follows ControlNet practice: the module
starts inert and can only move away from the baseline through training.

**`forward(x_t, t, text, labels, ctx=None, feats_low=None, return_feats=False)`:**

- A re-implementation of `UNet2DConditionModel.forward` with the same order of operations:
  - time and class embedding;
  - `conv_in` on `cat([x_t, ctx or zeros])`;
  - the down blocks, collecting residuals;
  - the mid block;
  - the up blocks, with `CRF1` / `CRF2` applied to the outputs of `up_blocks[1]` / `up_blocks[2]` when
    `feats_low` is given;
  - `conv_norm_out`, `conv_act`, `conv_out`.
- With `return_feats=True` it also returns the two up-block outputs, so pass 1 can export `h_low`.
- Cheaper alternative: register forward hooks on `up_blocks[1]` and `up_blocks[2]` and call the stock
  forward. That also works and keeps the diffusers code untouched.

**Lower-bucket map `b_low`:** 12→8, 16→12, 20→16, 24→16. For 24 px, the paper found 16 better than the
nearest bucket, 20. Buckets ≥ 32 are trained without context; they are out of scope.

**Sampling (`guide_mode crsc`), per step:**

- (1) `e_low, h_low = net(x, t, c, b_low, ctx=0, return_feats=True)`, then
  `x0_low = clamp((x - sqrt(1 - abar) e_low) / sqrt(abar), -1, 1)`.
- (2) `e_t = net(x, t, c, b, ctx=x0_low, feats_low=h_low)`.
- (3) `e = e_low + w (e_t - e_low)`:
  - `w = 1` is "CRSC alone" (the PASS row);
  - `w > 1` is "CRSC + label guidance" (the STRONG row, still 2 NFE);
  - `--guide_ckpt` swaps step (3)'s reference for the snapshot at `b_low` (composed, 3 NFE);
  - `--cfg_text` adds a context-free unconditional branch (3 NFE).

### 7.3 Training (`train_crsc.py`)

**Initialisation:** v7h EMA through `from_v7h`. Exclude list: `runs_out/holdout_exclude.txt`. Data sources are
identical to v7h (BLIP captions, so numbers stay comparable with 12.49 / 7.53); the dataset gains the 8 px
rung. Replicate on v7r when it finishes.

**Per batch** (bucket `b`, from `BucketSampler`):

1. Text dropout of 10 %, as in v7h. Sample `t`, `eps`, and form `x_t`.
2. Context mask `m ~ Bernoulli(0.5)` per sample; `m = 0` when `b` is the lowest rung.
3. For `m = 1` (and `m & roll`):
   - *Rolled variant*, only if T1 says so: with probability 0.5 of the context subset, replace `x_t` by
     `x~_t`. Noise `x0` to the previous sampler timestep `t'` and run one EMA-network CRSC step (2 NFE,
     `w_train = 1`, no grad). The target becomes `eps~ = (x~_t - sqrt(abar_t) x0) / sqrt(1 - abar_t)`.
   - Then, under `torch.no_grad()` with the **online** weights (self-conditioning practice), compute
     `e_low, h_low` at `b_low` with context 0, and from them `x0_low`.
4. `pred = net(x_t, t, c, b, ctx = m * x0_low, feats_low = h_low where m)`; batch the two groups or loop over
   them.
5. `loss = mean ||pred - target||^2`. Log the loss separately for context and no-context samples.

**Hyperparameters:**

- AdamW; learning rate 5e-5 for the UNet and 1e-4 for the adapters and the new `conv_in` slice.
- EMA 0.999; 20 k steps; batch sizes from `BATCH`, same `bs_scale` as v7h.
- Snapshots every 5 k steps; resume from `ckpt.pt`, following `train_v7.py`.

**Logged diagnostics every 1 k steps, on a fixed held-out batch:**

- (i) MSE with context vs without context;
- (ii) the absorption slope from 3.2;
- (iii) the norm of `e_b - e_low` against v7h's, to catch the shrinking difference that signals absorption.

**Budget:** about 4.1 h (forward-noised recipe) or about 4.7 h (with rolled states) for the main run. Each
paired control costs about the same.

### 7.4 Evaluation

Matched FD-DINOv2 protocol, 3000 prompts:

- 16 px first, then 12 / 20 / 24; 3 seeds for headline rows.
- Also report mean/covariance split, precision/recall/coverage, q16, ncol/flat/TV, Inception FID, and
  CLIP / R@1 for text alignment.
- Every table gets an **NFE/step** column and a **snapshot stored?** column. The Table-i style timing is
  re-measured.

### 7.5 Ablations the paper needs

Priority order, given one contended GPU:

1. **Module on/off.** A continued-training control: v7h + 20 k plain steps, same data and learning rate.
   Mandatory, because 20 k extra steps alone moved bare FD from 21.98 to 19.17 in probe_cg.
2. **Context source**, all on the same trainer:
   - lower label (CRSC);
   - **same label** (self-conditioning on the same `x_t`, i.e. two-pass self-refinement), which is the
     novelty-critical control;
   - higher label (20 / 24);
   - text-dropped at the same label;
   - snapshot at the lower label;
   - zero.

   Prediction from the finding: lower > same ≈ zero > higher.
3. **Guidance on top**, with NFE shown: none (2) / CFG (3) / label, reusing pass 1 (2) / composed (3).
   Include the `w` sweep {1, 1.25, 1.5, 2}; the optimal `w` should drop, or the curve flatten, if CRSC has
   internalised part of the guidance.
4. **Internalisation baselines** at the same step budget:
   - Meng-style distillation of the label and composed rules (1 NFE);
   - MG-form (candidate 3);
   - GFT-form (candidate 1b);
   - IG-style late split (1a, the zero-training T3 row);
   - class-embedding extrapolation (T2, zero training).
5. **Fusion site:** input channel only / decoder adapters only / both.
6. **Training states:** forward-noised / plus rolled / **SOAR-alone** (rolled states, no context). The last
   is required if the rolled recipe is used.
7. **Lower-bucket choice per resolution:** nearest vs farther; 24 px with 16 vs 20; 12 px with the 8 px rung
   vs a 10 px rung.
8. **Mechanism figure:**
   - the absorption/repulsion slope against `t`;
   - TV/ncol/flat of CRSC samples;
   - whether CRSC's own `e_b - e_low` has the "low-frequency, radial, colour" signature the sampling paper
     found (APG, FDG and channel-split probes).

   The headline claim is available only if the slope is positive: "trained only on data, the network learns
   to extrapolate away from its own lower-resolution belief."
9. **Second model** (v7s, 16 px only) if the budget allows. Then the recaptioned v7r.

### 7.6 Decision tree and kill criteria

1. **Run T0-T3 (about 1.5 GPU-h).**
   - If T0 or T1 gives `w* ≥ 1.3`: go to step 2, using the recipe that matches (forward-noised or rolled).
   - If both are below 1.1: skip to step 5.
2. **CRSC smoke, 5 k steps (about 1 h).**
   - Continue if the context MSE gain is ≥ 1.0 % at `t/T ≤ 0.5` **and** the slope is > 0.1.
   - Otherwise go to step 5.
3. **CRSC main (20 k) plus controls 1 and 2 (same-label context).** About 12 h in total.
4. **Evaluate against the bars:**
   - PASS if CRSC `w = 1` at 2 NFE is < 11.8 (3 seeds, beyond seed noise of 12.49 ± 0.72) and below the
     continued-training control's best CFG.
   - STRONG if CRSC + label guidance at 2 NFE, no snapshot, is ≤ 7.3 (3 seeds; 7.53 ± 0.19).
   - Kill if CRSC `w = 1` is > 14 **and** CRSC + label is ≥ 8.5.
   - Novelty kill if same-label context comes within 0.5 FD of lower-label context: report the result as
     self-refinement and drop the counterfactual claim.
5. **Fallback:** candidate 1b (GFT form, composed reference used only in training), 10-20 k steps.
   - INTERNALISER claimed if 1 NFE is ≤ 7.8 over 3 seeds.
   - Write it up as "GFT with a cross-resolution reference", with Meng and MG rows beside it.

---

## 8. Caveats

- **Novelty.** "No prior art found" for CRSC comes from three targeted searches plus the verified
  neighbourhood (self-conditioning, IG, SSG, SGG, MG, GFT, NAG, Matryoshka). It is not proof of absence.
- **Sampling-paper exposure.** Han et al. (CVPR 2026) guide with a same-network, *semantically degraded
  condition*: content tokens degraded, used as the negative instead of the null prompt, at sampling time only.
  This is conceptually close to the label reference ("a degraded condition of the same network"). The draft's
  related work should add it next to ICG/TSG and SDXL micro-conditioning.
- **Estimates.** The MAC split is analytic, not profiled. The success probabilities are rough priors to size
  the bets, not measurements.
- **Unit mismatch.** The n=200 external-baseline FDs in `experiment_log.md` (floor 26.53) are not comparable
  to the n=3000 numbers used here.

---

## 9. Verified references (all fetched 2026-09-12)

**Read in full HTML or source:**

- Karras, Aittala, Kynkäänniemi, Lehtinen, Aila, Laine. *Guiding a Diffusion Model with a Bad Version of
  Itself.* NeurIPS 2024. https://arxiv.org/abs/2406.02507
- Chen, Jiang, Zheng, Chen, Su, Zhu. *Visual Generation Without Guidance* (GFT). ICML 2025.
  https://arxiv.org/abs/2501.15420. Eq. 6 and eq. 9 read in HTML.
- Tang, Bao, Chen, Guo. *Diffusion Models without Classifier-free Guidance* (MG). arXiv 2502.12154.
  https://arxiv.org/abs/2502.12154. Target `eps + w * sg(eps~(c) - eps~(null))` with an EMA network and
  1 NFE, read in HTML.
- Yuan, Huang, Lei, Zhao, Wang, Chi, Wang, Zhang. *Improving Diffusion Generalization with Weak-to-Strong
  Segmented Guidance* (SGG, incl. BR branch and training migration). arXiv 2603.20584.
  https://arxiv.org/abs/2603.20584. Training loss read in HTML.
- Zhou, Li, Hu, Chen, Gu. *Guiding a Diffusion Transformer with the Internal Dynamics of Itself* (IG).
  arXiv 2512.24176. https://arxiv.org/abs/2512.24176. One-pass `D_i + w (D_f - D_i)` read in HTML.
- Pokle, Galashov, Doucet, Delbracio, De Bortoli. *Adversarial Learning of Classifier-Free Guidance
  Schedules.* arXiv 2608.14038. https://arxiv.org/abs/2608.14038. Scalar MLP guidance, frozen base, read in
  HTML.
- Chen, Zhang, Hinton. *Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning.*
  ICLR 2023. https://arxiv.org/abs/2208.04202. Self-conditioning details read on ar5iv.
- diffusers `UNet2DConditionModel` source (class embedding added to the time embedding;
  `resnet_time_scale_shift="default"`).
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_condition.py

**Read at abstract level only:**

- Meng, Rombach, Gao, Kingma, Ermon, Ho, Salimans. *On Distillation of Guided Diffusion Models.* CVPR 2023.
  https://arxiv.org/abs/2210.03142
- Podell et al. *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* 2023.
  https://arxiv.org/abs/2307.01952. The abstract mentions "multiple novel conditioning schemes".
- Ahn et al. *Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance.* ECCV 2024.
  https://arxiv.org/abs/2403.17377
- Gu, Zhai, Zhang, Susskind, Jaitly. *Matryoshka Diffusion Models.* ICLR 2024.
  https://arxiv.org/abs/2310.15111
- Fu, Wang, Guo, Zhou, Nie, Wen. *A Frozen Pixel-Space Diffusion Model Can Guide Itself with Its Own
  Samples* (SSG). arXiv 2607.29122. https://arxiv.org/abs/2607.29122
- Galashov, Pokle, Doucet, Gretton, Delbracio, De Bortoli. *Learn to Guide Your Diffusion Model.*
  arXiv 2510.00815. https://arxiv.org/abs/2510.00815
- Chen, Su, Sun, Zhu. *Toward Guidance-Free AR Visual Generation via Condition Contrastive Alignment* (CCA).
  arXiv 2410.09347. https://arxiv.org/abs/2410.09347
- Zheng et al. *Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is
  Secretly a GAN Discriminator.* ICML 2025. https://arxiv.org/abs/2503.01103
- Kim, Kim, Kwon, Kang, Moon. *Refining Generative Process with Discriminator Guidance in Score-based
  Diffusion Models.* ICML 2023. https://arxiv.org/abs/2211.17091
- Ning, Sangineto, Porrello, Calderara, Cucchiara. *Input Perturbation Reduces Exposure Bias in Diffusion
  Models.* ICML 2023. https://arxiv.org/abs/2301.11706
- Ning, Li, Su, Salah, Ertugrul. *Elucidating the Exposure Bias in Diffusion Models* (epsilon scaling).
  ICLR 2024. https://arxiv.org/abs/2308.15321
- Qin et al. *SOAR: Self-Correction for Optimal Alignment and Refinement in Diffusion Models.*
  arXiv 2604.12617. https://arxiv.org/abs/2604.12617
- Shen, Song, Xue, Wang, Liu. *Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance*
  (S-CFG). CVPR 2024. https://arxiv.org/abs/2404.05384
- Li et al. *Delta Score Matters! Spatial Adaptive Multi Guidance in Diffusion Models* (SAMG).
  arXiv 2604.26503. https://arxiv.org/abs/2604.26503
- Han, Zhang, Wang. *Guiding Diffusion Models with Semantically Degraded Conditions.* CVPR 2026.
  https://arxiv.org/abs/2603.10780
- Hsiao et al. *Plug-and-Play Diffusion Distillation.* CVPR 2024. https://arxiv.org/abs/2406.01954
- Sadat, Kansy, Hilliges, Weber. *No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion
  Models* (ICG/TSG). ICLR 2025. https://arxiv.org/abs/2407.02687
- Li, Luo, Chen, Ma, Qi. *Self-Guidance: Boosting Flow and Diffusion Generation on Their Own.*
  arXiv 2412.05827. https://arxiv.org/abs/2412.05827
- Chen, Bandyopadhyay, Zou, Song. *Normalized Attention Guidance: Universal Negative Guidance for Diffusion
  Models* (NAG). arXiv 2505.21179. https://arxiv.org/abs/2505.21179
