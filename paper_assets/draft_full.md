<!-- Title choice: candidate 1 from paper_outline.md §1 ("Cross-Resolution Self-Guidance: A Diffusion Model's Lower-Resolution Belief Is Its Own Best Bad Model"); candidates 2/3 name only the pixel-art setting or the "contrast-deficient" reading, which §5 now presents as a characterisation rather than a proven mechanism. -->

# Cross-Resolution Self-Guidance: A Diffusion Model's Lower-Resolution Belief Is Its Own Best Bad Model

## Abstract

Diffusion models of very-low-resolution pixel art (12–32 px RGBA sprites) regress towards the mean, producing bleeding colours and insufficient local contrast, and classifier-free guidance (CFG) trades this error against text alignment along a single frontier. We show that a *bucketed* multi-resolution model already contains the weak reference that autoguidance needs: the same weights, noisy input and caption with the label of a *lower* resolution bucket yield a structure-aligned prediction with systematically lower local contrast, and extrapolating the strong prediction away from it removes the error with no training at the cost of CFG. Against the *best* CFG weight at each resolution (not the training default), composing this reference with an early snapshot lowers FD-DINOv2 by 40 / 27 / 29 / 17 % at 16 / 20 / 24 / 32 px (16 px: 12.5 → 7.5, floor 3.45; 3 seeds) and by 45 / 35 / 36 % on a second, narrower model, at equal or better CLIP alignment than the CFG optimum at 16 px, whereas perturbed-attention guidance matches CFG at every weight. Higher-bucket references hurt everywhere, and a characterisation of the weak references with one interventional control shows that an effective reference has lower total variation than the strong prediction while remaining structure-aligned: uniformly shrinking the strong prediction's own contrast is catastrophic.

## 1 Introduction

Pixel-art sprites are among the smallest images anyone draws on purpose: a character or item on a 12–32 px RGBA canvas, made of hard edges, a few flat colour regions and a transparent background. There is no texture in which to hide a mistake, and the perceived quality of a sprite is carried almost entirely by per-pixel colour statistics and local contrast. The standard metric sees the same thing: a 16 px sprite upsampled with nearest-neighbour interpolation to the 224 px input of DINOv2 [cite: Oquab2023] maps every 14 × 14 patch to one sprite pixel, so the Fréchet distance in DINOv2 space (FD-DINOv2, [cite: Stein2023]) penalises isolated wrong pixels and mixed colours heavily.

Measured this way, a well-trained text-to-sprite diffusion model is far from the data. Our clean 16 px baseline (a standard RGBA UNet with a frozen CLIP text encoder and a resolution-bucket class embedding, trained from scratch with the evaluation sprites excluded) reaches FD-DINOv2 21.52 ± 0.47 (three seeds) under CFG [cite: HoSalimans2022] at the training-recipe default w = 4, whereas a second set of held-out real sprites scores 3.45. The default is over-guided: sweeping w over {1, 1.5, 2, 3, 4, 7, 10} gives a U-shaped curve with its minimum at w = 1.5, 12.49 ± 0.72, and no CFG weight reaches below 12.2 (Appendix Table A1). We therefore report every gain against this *best-CFG* baseline, re-tuned at each resolution, and give the w = 4 figure second. Even at its optimum CFG leaves a large systematic error — about half of the remaining FD (mean term 5.87 of 12.24, seed 0) is a shift of the feature mean, against 0.38 for real sprites — visible as colour bleeding: adjacent colours mix and local contrast is lost. Changing how CFG is applied does not repair it: CADS [cite: Sadat2024cads] and a guidance interval [cite: Kynkaanniemi2024] are worse than plain CFG, and perturbed-attention guidance [cite: Ahn2024] tracks the CFG curve (12.57–12.90 for w = 1.5–3). This is the regime autoguidance [cite: Karras2024] targets by extrapolating away from a deliberately weaker, "compatibly degraded" version of the same model, and it works here: an EMA snapshot from step 10 k of the same run lowers FD to 8.73 ± 0.26, 30 % below best-CFG. It requires, however, a second set of weights to store and, more generally, a snapshot that may not exist.

Our observation is that a *bucketed* multi-resolution model already contains a compatible weak reference. Such models receive the target resolution as a label; ours is trained over buckets {12, 16, 20, 24, 32, 48, 64} px, each sprite also seen box-downsampled at the lower buckets. Given the same weights, the same noisy input x_t and the same caption but the label of a *lower* bucket (12 when generating at 16), the output is not a smaller sprite: it is a prediction on the same 16 × 16 grid with the same layout and about the same number of colours as the strong prediction (78 vs 73 median unique colours) but systematically lower local contrast — mean adjacent-pixel difference (total variation, TV) 21.6 against 32.0 for the strong prediction and 30.2 for real sprites. We call it the model's *lower-resolution belief*: a structure-aligned, lower-contrast version of what the model predicts, which is the direction of the error we want to remove.

The method is a single line. For target bucket b_R and caption c, compute e_strong = ε_θ(x_t, t, c, b_R) and e_weak = ε_θ(x_t, t, c, b_low) with b_low < b_R, and denoise with e = e_weak + w (e_strong − e_weak). This *replaces* CFG, so each step still costs two network evaluations; no unconditional forward, no second model, no training. The weak forward can also be taken through an early snapshot under the lower label, composing both references at unchanged cost; this *composed* reference is the method we advertise, and the label alone is its free, zero-storage component. Network, data and training recipe are standard and unchanged; this paper contributes a guidance rule and its analysis, not an architecture.

**Contributions.**

- **A zero-training guidance rule for bucketed diffusion models, and its composition with autoguidance.** The composed reference lowers 16 px FD-DINOv2 to 7.53 ± 0.19 over three seeds, 40 % below the best CFG weight (65 % below the training default) and below every seed of either single reference; the label-only reference reaches 8.52 ± 0.29 (−32 %), matching early-snapshot autoguidance (8.73 ± 0.26, −30 %) with no snapshot stored, at the wall-clock and memory of CFG (§4).
- **Replication and robustness.** Against best-CFG re-tuned at each resolution the composed reference gains −27 / −29 / −17 % at 20 / 24 / 32 px and −45 / −35 / −36 % at 16 / 20 / 24 px on a second, narrower model of the same family; it is best or tied-best under Inception clean-FID/KID at every resolution, with smaller margins (−8 to −14 %). The label-only gain shrinks with resolution (−17 / −12 / −7 %), so the free variant is a 16 px result plus a component of the composed method (§4.4–4.7).
- **Directionality and an alignment–fidelity frontier.** A *higher*-bucket reference is worse than best-CFG at every resolution and worse than the w = 4 default at 12, 20, 24 and 32 px. On the FD–CLIP plane the guided rows lie below the CFG curve, and a variant that folds the empty caption into the weak branch recovers real-data-level CLIP at +0.9 FD (§4.6).
- **A characterisation with one interventional control.** At three resolutions every reference that helps has lower TV than the strong prediction while keeping its colour count and structure; every reference with TV at or above it is ineffective or harmful. Low TV is not sufficient — a trained block-average branch is harmful, a trained contrast-shrunk branch weak — and the reference is not a low-contrast *scalar*: shrinking the strong prediction's own contrast is monotonically harmful (§5).

## 2 Related Work

**Guidance in diffusion models.** CFG [cite: HoSalimans2022] extrapolates the conditional prediction away from an unconditional one from the same network. Its failure modes have motivated variants that keep the unconditional reference but change how the difference is applied: guidance intervals [cite: Kynkaanniemi2024], APG [cite: Sadat2024apg] (removes the component parallel to the prediction, adds momentum), FDG [cite: Sabour2025] (down-weights low-frequency bands), CADS [cite: Sadat2024cads], feedback and time-varying schedules [cite? Koulischer2025; cite? TVCFG2025] and CFG++ [cite: Chung2024]. Our rule changes the *reference*, not the weight or the application of the difference; §5.3 reports that APG, FDG, interval scheduling and CADS do not improve on it in our setting.

**Weak-model and self references.** Autoguidance [cite: Karras2024] replaces the unconditional reference by a smaller or less-trained version of the same model, arguing that the weak model must share the strong model's errors in amplified form (a "compatible degradation"); its ablations vary capacity and training time, not the conditioning label. It is our control and one half of the composed reference. Other works obtain a weak prediction without a second model: blurring attention-selected input regions [cite: Hong2023], crops that restrict the receptive field [cite? SWG2024], sampled sub-networks [cite? S2Guidance2025], inference-time dropout (reported ineffective, [cite? InSituAG2025]), a random condition or perturbed time-step embedding (ICG/TSG, [cite: Sadat2024icg]), a degraded *text* condition (CDG, [cite? CDG2026]), or jointly trained intermediate-layer heads as low-frequency predictors [cite? SGGBR2026; cite? IG2025; cite? SSG2026]. Ours changes only the target-resolution label: same network, same input, same text. Degraded-input references fail in our regime (2 × 2 box-blurred x_t: FD 223.07; 1 px shift: 18.95), a trained block-average branch — the 16 px analogue of a low-frequency head — is harmful (§5.4), and the reference is not random: a lower label helps, a higher label hurts (§4.4), and mixing the lower label with the unconditional prediction lands between the two (9.42 vs 8.59 and 21.98, seed 0).

**Perturbed self-attention.** PAG [cite: Ahn2024] and SEG [cite: Hong2024] obtain a training-free weak prediction by replacing or blurring the self-attention map inside the network at the same 2 NFE as CFG. They perturb the *computation*; we perturb only the *condition*. PAG is our direct zero-training competitor (§4.3): in our setting it tracks the CFG curve — 12.57 / 12.62 / 12.90 at w = 1.5 / 2 / 3 against best-CFG 12.24 — pays the same CLIP, and does not remove the systematic error that the lower-bucket reference removes (8.59, same seed).

**SDXL micro-conditioning.** The closest prior trick is SDXL's size conditioning [cite: Podell2023]: in common practice [cite? diffusers] users pass a small `negative_original_size` to the negative branch of CFG, documented to induce "simpler patterns". There the reference also drops the text, the size label encodes the quality of the *source* image rather than the target bucket, and the effect has, to our knowledge, not been quantified or tested for direction. Our `bucketu:12` rows (lower bucket *and* empty caption in the weak branch, Appendix Table A3) are its analogue in our model: 1.9–3.5 FD worse than keeping the caption fixed (10.45 / 12.10 vs 8.59, seed 0), and at w = 2 no better than the best CFG point (12.24).

**Multi-resolution diffusion and pixel art.** Training one model over several resolutions with a bucket label is standard, from aspect-ratio bucketing [cite? NovelAI2022] and SDXL to Matryoshka diffusion [cite: Gu2023], FiT [cite: Lu2024], ScaleCrafter [cite: He2023] and simple diffusion [cite: Hoogeboom2023]; none uses the embedding as a guidance reference. Learned pixel-art work mostly converts images to pixel art [cite: Gerstner2012; cite: Han2018; cite: Wu2022; cite? PixelOE2024]; generative sprite models include GANs [cite: Coutinho2022], palette-conditioned diffusion [cite? PixDiffPIG2025; cite? SpriteSheetDiffusion2024] and SD-πXL [cite: Binninger2024], which optimises a palette assignment per image by score distillation. Colour bleeding is routinely handled by palette quantisation after generation; we report 16-colour quantisation as a secondary column (§4.2). We are not aware of prior work on guidance for diffusion at or below 32 px.

**Evaluating tiny images.** FID [cite: Heusel2017] and clean-FID [cite: Parmar2022] compute Inception features on natural-image-sized inputs; Stein et al. [cite: Stein2023] argue for DINOv2 features, under which a 16 px sprite pixel maps to exactly one patch. We use FD-DINOv2 with a matched protocol as the primary metric, with its mean/covariance decomposition, precision/recall [cite: Kynkaanniemi2019] and density/coverage [cite: Naeem2020] as diagnostics, Inception clean-FID and KID [cite: Binkowski2018] on the same samples as a second family (§4.7), and CLIP similarity [cite: Radford2021] for text faithfulness (§4.6).

## 3 Method

### 3.1 Problem setup

We generate RGBA sprites at a target side length $R \in \{12, 16, 20, 24, 32\}$ px from a caption $c$. The generator is a standard conditional diffusion model and is not a contribution: a 4-channel RGBA UNet (`UNet2DConditionModel`, 72.5 M parameters (72,497,540 in the EMA state dict)) with cross-attention to a frozen CLIP text encoder [TODO cite CLIP], and a *resolution bucket* $b \in \mathcal{B} = \{12, 16, 20, 24, 32, 48, 64\}$ supplied as a class embedding. It is trained with $\epsilon$-prediction [TODO cite Ho et al. 2020], sampled with 100 DDPM steps and evaluated with EMA weights. Training data are OGA-derived sprites with BLIP captions [TODO cite BLIP]; each sprite is also fed, BOX-downsampled, to every lower bucket, so all buckets see the same sprite population, and the training set is 16 px-dominated. The main model (v7h) is trained from random initialisation for 80 k steps on 180,533 rows with the 5,928 evaluation sprites excluded; EMA snapshots are saved every 5 k steps. A second model (v7s; width 96, 41.3 M parameters, 60 k steps, different seed) shares recipe and exclusion list.

Write $\epsilon_\theta(x_t, t, c, b)$ for the noise prediction at noisy input $x_t$, timestep $t$, caption $c$ and bucket $b$. The baseline is CFG with the empty caption $\varnothing$:

$$
e_{\text{cfg}} = \epsilon_\theta(x_t,t,\varnothing,b_R) + w_{\text{cfg}}\big[\epsilon_\theta(x_t,t,c,b_R) - \epsilon_\theta(x_t,t,\varnothing,b_R)\big].
$$

The training-recipe default $w_{\text{cfg}} = 4$ gives FD-DINOv2 $21.52 \pm 0.47$ (3 seeds) at 16 px against a real-data floor of $3.45$, but it is over-guided: over $w_{\text{cfg}} \in \{1, 1.5, 2, 3, 4, 7, 10\}$ the FD curve is U-shaped with its minimum at $w_{\text{cfg}} = 1.5$ ($12.49 \pm 0.72$, 3 seeds), and CADS and interval variants lie above the curve (Appendix Table A1). We therefore use *best-CFG* — the weight with the lowest FD at each resolution: 1.5 at 16 px, 2.5 at 20 px, 2 at 12 / 24 / 32 px — as the baseline throughout, and keep $w_{\text{cfg}} = 4$ as the "training default" row. Even at its optimum CFG leaves a mean term of 5.87 (seed 0; floor 0.38), and no CFG weight reaches FD < 12.2. The gap is a systematic model error, the regime that autoguidance [cite: Karras2024] targets by extrapolating away from a *worse but compatible* version of the same model.

### 3.2 Cross-resolution self-guidance

Every guidance rule we consider has the form: a strong prediction $e_{\text{strong}} = \epsilon_\theta(x_t,t,c,b_R)$, a weak reference $e_{\text{ref}}$ on the same $x_t, t$, and

$$
\boxed{\; e = e_{\text{ref}} + w\,\big(e_{\text{strong}} - e_{\text{ref}}\big) \;} \qquad \text{(2 network evaluations per step).}
$$

CFG is $e_{\text{ref}} = \epsilon_\theta(x_t,t,\varnothing,b_R)$; autoguidance is $e_{\text{ref}} = \epsilon_{\theta_{\text{snap}}}(x_t,t,c,b_R)$ with $\theta_{\text{snap}}$ an early EMA snapshot of the same run (step 10 k, $w = 1.5$).

Our observation is that a bucketed model already contains a compatible weak model without any second set of weights: the same network, on the same $x_t$ and caption, queried under a **lower** bucket label $b_{\text{low}} < b_R$,

$$
e_{\text{ref}} = \epsilon_\theta(x_t,t,c,b_{\text{low}}), \qquad w = 2 .
$$

The rule replaces CFG — no unconditional forward is computed — so its cost is that of CFG. At 16 px with $b_{\text{low}} = 12$ it gives $8.52 \pm 0.29$ (3 seeds), within seed noise of snapshot autoguidance ($8.73 \pm 0.26$) and 32 % below best-CFG, with no snapshot stored (Table 1). Sampled alone, the lower-resolution belief is a clearly worse model (FD 36.43 vs 21.98 at 16 px, seed 0) — autoguidance's "bad model" premise — and it is structure-aligned with the strong prediction by construction, since only the label differs.

**Choice of $b_{\text{low}}$.** The nearest lower bucket is the default (12 for 16 px, 16 for 20 px, 24 for 32 px). At 20 px the nearest bucket beats the farther one (32.75 ± 0.60 over three seeds vs 35.85, seed 0, for bucket 16 vs 12); at 32 px the farther bucket is clearly worse (91.26 vs 77.45, seed 0). The exception is 24 px, where the nearest lower bucket (20) is the worst choice (66.20, seed 0) and we use bucket:16 (59.32 ± 1.37 over three seeds), while bucket:12 (57.70, seed 0) is marginally better still; this choice was made on the reported FD (§6). At 12 px no lower bucket exists and the method does not apply; only snapshot autoguidance is available. An operational rule for choosing the reference in a new model is given in Appendix A.15.

**Choice of $w$.** The label reference's $w$-curve is flat (10.21 / 8.59 / 11.20 / 13.87 for $w = 1.5, 2, 2.5, 3$ at 16 px), the snapshot's steep (8.98 → 30.11 for $w = 1.5 \to 3$), and the same shape holds at 20 and 24 px (worst-over-sweep / best 1.25× and 1.23× for the label reference against 1.86× and 1.59× for the snapshot, $w \in [1.25, 3]$; Appendix Table A1). We therefore use one fixed $w$ per reference type at every resolution and on both models: $w = 2$ for the label reference and $w = 1.5$ whenever the snapshot is involved. CFG, by contrast, needs per-resolution tuning (best weight 1.5 / 2.5 / 2 / 2 at 16 / 20 / 24 / 32 px).

### 3.3 Composed reference

The two weakening axes — earlier weights and a lower label — can be applied in a single forward pass. The composed reference evaluates the snapshot under the lower label:

$$
\boxed{\; e^{\text{comp}}_{\text{ref}} = \epsilon_{\theta_{\text{snap}}}(x_t,t,c,b_{\text{low}}), \qquad e = e^{\text{comp}}_{\text{ref}} + w\,\big(\epsilon_\theta(x_t,t,c,b_R) - e^{\text{comp}}_{\text{ref}}\big), \quad w = 1.5 . \;}
$$

This is one weak reference, not a sum of two guidance terms; the sampler still performs two evaluations per step. At 16 px it reaches $7.53 \pm 0.19$, 40 % below best-CFG and below every seed of either single reference, and the ordering bare $>$ label reference $\approx$ autoguidance $>$ composed holds at every resolution with a lower bucket (16 / 20 / 24 / 32 px, Table 2) and on the second model (Table 3). The composed reference is the method we advertise. The label alone matches autoguidance at 16 px, but its stand-alone gain over best-CFG shrinks with resolution (−32 % at 16 px, −17 / −12 / −7 % at 20 / 24 / 32 px; autoguidance −30 / −20 / −20 / −10 %), whereas the composed gain stays at −40 / −27 / −29 / −17 %; the label is best read as the free component that removes the mean shift (§5.3).

The two references are complementary (Appendix Table A6, seed 0). Splitting FD into mean and covariance terms, the label reference has the lowest mean term (2.55 vs 3.88 for autoguidance and 5.87 for best-CFG) but weaker covariance term and coverage (6.04, .892); the snapshot has the best covariance term and coverage (5.09, .928); the composed reference keeps the label's mean term (2.40) and most of the snapshot's covariance gain (5.27; coverage .909). The snapshot matters only in the low-noise half: using its weights only for $t/T \le 0.5$ ("snaplo"; outside the interval the reference is the final weights under $b_{\text{low}}$) gives $7.63 \pm 0.43$, indistinguishable from the full rule, while $[0.5, 1]$ gives 8.99, close to the label reference alone (§5.3). Snaplo is a diagnostic, not a saving: evaluations per step are unchanged.

**Cost.** For 1000 samples at 16 px on one A100-80GB (Appendix Table A9) the label reference costs what CFG costs (72.7 vs 73.8 s, identical 4951 MiB peak memory); snapshot-based references hold one extra copy of the weights (+556 MiB) at the same wall-clock (73.3–74.0 s).

## 4 Experiments

### 4.1 Setup

**Held-out protocol.** Every experiment uses v7h (§3.1) unless stated. A fixed set of 5,928 evaluation sprites is excluded from training. We generate one sample for each of 3,000 held-out captions (seed 0 unless stated) and compare against 3,000 real sprites passed through the same `to_tensor(R)` preprocessing as training (aspect-preserving, centred on a transparent canvas, hard alpha). The primary metric is *matched* Fréchet distance in DINOv2-small CLS feature space (FD-DINOv2; lower is better) at the native resolution $R$, nearest-neighbour upsampled to the 224 px input. Because the reference set is finite and disjoint from the held-out sprites the metric has a non-zero floor: the held-out real sprites themselves score **3.37 / 3.45 / 12.14 / 12.96 / 13.77** at 12 / 16 / 20 / 24 / 32 px. Where indicated we also report FD after 16-colour quantisation of the samples (+q16), a diagnostic that separates colour-domain from structural effects. Seed-to-seed sd is 0.2–0.5 FD at 16 px; headline rows use three seeds (0, 1, 2), and differences of the order of the seed spread are not interpreted.

**Rows.** *Best-CFG* is the CFG weight with the lowest FD at each resolution ($w = 1.5$ at 16 px, 2.5 at 20 px, 2 at 12 / 24 / 32 px for v7h; swept over $\{1, 1.5, 2, 2.5, 3, 4, 7, 10\}$ at 16 px and $\{1.5, 2, 2.5, 3, 4\}$ elsewhere), re-run with three seeds wherever it serves as a 3-seed baseline; the $w = 4$ row is the training default. Percentages are given against best-CFG first and against $w = 4$ in parentheses. The *label reference* uses the same weights, $x_t$ and caption with a lower bucket label, $w = 2$; *autoguidance* [cite: Karras2024] uses the step-10 k snapshot under the correct label, $w = 1.5$; the *composed* reference evaluates the snapshot under the lower label, $w = 1.5$. All cost two network evaluations (NFE) per step, like CFG. Controls: CADS, interval-restricted CFG and PAG [cite: Ahn2024]; *reverse* references with a *higher* bucket label; `bucketmix:12` (half lower bucket, half unconditional); two *trained* degraded-view branches (`probe_cg`, `probe_cc`) and one untrained *interventional* control, `shrink:f` (§5.4).

### 4.2 Main results at 16 px

Table 1 reports the 16 px results over three seeds. CFG at the training default is far from the floor (21.52 vs 3.45); re-tuning its weight to the FD optimum $w = 1.5$ brings it to 12.49 ± 0.72 (seeds 12.24 / 13.32 / 11.91), and no CFG weight goes below 12.2 (§4.3). Replacing the unconditional reference with the model's own lower-bucket belief gives 8.52 ± 0.29 with no training or stored weights, 32 % below best-CFG (60 % below $w = 4$), matching autoguidance with a stored snapshot (8.73 ± 0.26, −30 %; the 0.2 difference is within seed spread). Composing both references in one weak forward gives 7.53 ± 0.19, 40 % below best-CFG (65 % below $w = 4$), below every single-reference seed and 4.6 seed-sd below autoguidance. Restricting the snapshot to the low-noise half gives 7.63 ± 0.43, indistinguishable from the full composition. After 16-colour quantisation autoguidance and the composed reference swap order (7.8 vs 8.3–8.4), so part of the composed gain lives in the colour domain (§4.7, §6).

**Table 1.** Matched FD-DINOv2 @16 px, v7h, 3,000 held-out captions, three seeds. Floor (held-out real sprites) = 3.45. +q16: FD after 16-colour quantisation (seeds 0 / 1 / 2). Best-CFG = the CFG weight with the lowest FD ($w = 1.5$, sweep in Appendix Table A1); the $w = 4$ row is the training-recipe default.

| Method | Weak reference | Extra training / storage | $w$ | seed 0 | seed 1 | seed 2 | FD mean ± sd | +q16 |
|---|---|---|---|---|---|---|---|---|
| Real held-out (floor) | — | — | — | 3.45 | — | — | **3.45** | — |
| CFG, training default | unconditional | none | 4 | 21.98 | 21.05 | 21.54 | 21.52 ± 0.47 | 12.64 / 12.68 / 12.55 |
| CFG, best weight (baseline) | unconditional | none | 1.5 | 12.24 | 13.32 | 11.91 | 12.49 ± 0.72 | 11.55 / — / — |
| Autoguidance | EMA snapshot 10 k | stores snapshot | 1.5 | 8.98 | 8.73 | 8.47 | 8.73 ± 0.26 | 7.82 / 7.98 / 7.82 |
| Label reference (ours) | `bucket:12`, same weights | **none** | 2 | 8.59 | 8.20 | 8.76 | 8.52 ± 0.29 | 8.99 / 8.78 / 8.57 |
| Composed (ours) | snapshot 10 k under `bucket:12` | stores snapshot | 1.5 | 7.67 | 7.32 | 7.61 | **7.53 ± 0.19** | 8.34 / 8.02 / 8.42 |
| Composed, snapshot only for $t/T\in[0,0.5]$ | as above; final weights elsewhere | stores snapshot | 1.5 | 7.70 | 7.16 | 8.02 | 7.63 ± 0.43 | [TBD] / 8.16 / 8.44 |

**Figure 1.** Prompt-aligned 16 px samples (`fig_qual_16px.png`; same seed and caption per column, FD in parentheses, seed 0). Rows: real held-out sprites; CFG $w = 4$ (21.98); best-CFG $w = 1.5$ (12.24); autoguidance 10 k, $w = 1.5$ (8.98); `bucket:12` self-guidance, $w = 2$ (8.59); composed, $w = 1.5$ (7.67); composed∘`bucketu:12`, $w = 1.5$ (8.60); the `bucket:12` belief sampled alone (36.43); the reverse `bucket:24` reference, $w = 2$ (17.27). The bare model's characteristic failures are bleeding colours and insufficient local contrast, which the guided rows reduce; the pure lower-bucket belief is a soft, low-contrast rendering of the same layout. Other resolutions: Appendix Figures A2–A5.

### 4.3 Guidance-weight sweeps and zero-training baselines

Appendix Table A1 collects the CFG weight curves at every resolution, the other zero-training CFG variants at 16 px and the sweeps of both single references at 16 / 20 / 24 px (seed 0). Three facts carry into the main text. (i) *CFG is over-guided at its default and cannot be re-tuned into the guided rows' range.* At 16 px the CFG curve is 16.39 / 12.24 / 12.98 / 14.56 / 16.67 / 21.98 / 42.95 / 62.12 for $w = 1 / 1.5 / 2 / 2.5 / 3 / 4 / 7 / 10$: the default is over-guided by 72 % and re-tuning removes 42 % of its FD, but every reference-guided row in Table 1 is below 9. CADS (44.58–88.61) and interval-restricted CFG (32.57) are worse than plain CFG at the same weight. The optimum shifts with resolution ($w = 2.5$ at 20 px, 2 at 12 / 24 / 32 px) and the over-guidance of $w = 4$ is much smaller there (re-tuning gains −16 / −14 / −14 % at 20 / 24 / 32 px vs −42 % at 16 px). (ii) *PAG* (identity self-attention in the mid block) gives 12.57 / 12.62 / 12.90 at $w = 1.5 / 2 / 3$, 13.05 with mid + `down_blocks.1`, and 11.73 with an added CFG term of 1.5 (3 NFE): a gentler CFG that tracks the CFG optimum without removing the systematic error. (iii) *Autoguidance's snapshot step matters little* (5 k / 10 k / 20 k / 40 k at $w = 2$: 14.89 / 11.33 / 12.14 / 11.62) whereas its weight matters a lot (§3.2); degraded-*input* references fail outright (box-blurred $x_t$: 223.07; 1-px cyclic shift: 18.95).

### 4.4 Generalisation across resolutions and reverse controls

Table 2 repeats the comparison at 12, 20, 24 and 32 px with the same weights and snapshot, against best-CFG re-tuned at each resolution and against the $w = 4$ default. The ordering bare > label reference ≈ autoguidance > composed holds at every resolution with a lower bucket (at 32 px autoguidance, 75.23, edges the label reference, 77.45, seed 0). Relative to best-CFG the composed reference gains −40 / −27 / −29 / −17 % at 16 / 20 / 24 / 32 px (−65 / −39 / −39 / −28 % relative to $w = 4$); where three seeds exist the gap is 6.9 / 15.5 / 25 times the larger seed sd at 16 / 20 / 24 px, and the composed − autoguidance gap is 4.6 / 3.9 / 8.2 sd. The *label-only* reference gains −32 % at 16 px but only −17 / −12 / −7 % at 20 / 24 / 32 px (autoguidance −30 / −20 / −20 / −10 %): the free variant matches autoguidance at 16 px and remains a component of the composed method, but its stand-alone advantage over re-tuned CFG shrinks with resolution. At 12 px, which has no lower bucket, autoguidance gives 8.54 ± 0.73 against best-CFG 9.71 (−12 %; −34 % vs $w = 4$). The bare model is much further from the floor at 20 / 24 / 32 px (best-CFG gaps 27.4 / 54.8 / 69.9 vs 9.0 at 16 px), reflecting the 16 px-dominated training data: the gain is largest where the model is weakest.

*Reverse controls.* If a wrong label merely acted as a generic unconditional-like reference, a higher bucket would help as much as a lower one. It does not (last column of Table 2, seed 0; full table in Appendix Table A4). Against best-CFG every reverse row is worse at every resolution (+4.7 at 12 px; +4.9 / +5.0 / +7.1 at 16 px; +7.1 / +19.6 at 20 px; +34.3 at 24 px; +30.3 at 32 px), and at 12, 20 (`bucket:32`), 24 and 32 px it is worse even than the $w = 4$ default (+1.5 / +13.2 / +21.9 / +17.1 against the seed-0 rows 12.91 / 45.92 / 79.80 / 96.85); at 20 px `bucket:24` is indistinguishable from $w = 4$ (46.62 vs 45.92). At 16 px higher buckets beat the $w = 4$ default in raw FD (17.14–19.34 vs 21.98) but not after quantisation (q16 18.64 / 21.02 / 24.56 vs 12.64), with precision rising to .935 and recall falling to .85–.88 (bare .907 / .902) — the signature of over-guidance contraction; v7s shows the same (`bucket:20`: 25.31 vs best-CFG 19.10; q16 24.14 vs 13.60; precision .925, recall .848). `bucketmix:12` (9.42) lies between the label reference (8.59) and best-CFG (12.24), not at CFG. The effect is ordered by the label.

**Table 2.** Matched FD-DINOv2 at native resolution, v7h. Mean ± sd over three seeds where marked (3 s); otherwise seed 0. "Best CFG" = the CFG weight with the lowest FD at that resolution (weight in parentheses). "Lower bucket used" = the nearest lower bucket except at 24 px, where bk16 is used and the nearest (bk20) is listed under "Other lower bucket", $w = 2$. Reverse = higher-bucket reference, $w = 2$, seed 0. Δ columns are relative to best-CFG (composed vs $w = 4$ in parentheses).

| $R$ | Floor | CFG $w=4$ (default) | Best CFG ($w$) | Lower bucket used, $w=2$ | Other lower bucket, $w=2$ | Autoguidance 10 k, $w=1.5$ | Composed, $w=1.5$ | Reverse (higher bucket) | Δ vs best CFG: label / autog. / composed (composed vs $w=4$) |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 3.37 | 13.02 ± 0.28 (3 s) | 9.71 (2) | — (none exists) | — | **8.54 ± 0.73** (3 s) | — | bk16 14.45 | — / −12 % / — (autog. −34 %) |
| 16 | 3.45 | 21.52 ± 0.47 (3 s) | 12.49 ± 0.72 (1.5; 3 s) | bk12 8.52 ± 0.29 (3 s) | — | 8.73 ± 0.26 (3 s) | **7.53 ± 0.19** (3 s) | bk20 17.14; bk24 17.27; bk64 19.34 | −32 / −30 / **−40 %** (−65 %) |
| 20 | 12.14 | 47.04 ± 1.11 (3 s) | 39.57 ± 0.04 (2.5; 3 s) | bk16 32.75 ± 0.60 (3 s) | bk12 35.85 | 31.57 ± 0.29 (3 s) | **28.90 ± 0.69** (3 s) | bk24 46.62; bk32 59.13 | −17 / −20 / **−27 %** (−39 %) |
| 24 | 12.96 | 78.96 ± 0.74 (3 s) | 67.77 ± 0.77 (2; 3 s) | bk16 (not nearest) 59.32 ± 1.37 (3 s) | bk20 (nearest) 66.20; bk12 57.70 | 54.15 ± 0.72 (3 s) | **48.23 ± 0.43** (3 s; bk16 + 10 k) | bk32 101.67 | −12 / −20 / **−29 %** (−39 %) |
| 32 | 13.77 | 96.85 | 83.65 (2) | bk24 77.45 | bk16 91.26 | 75.23 | **69.27** (bk24 + 10 k) | bk48 113.93 | −7 / −10 / **−17 %** (−28 %) |

### 4.5 A second, independently trained model

To check that the effect is not specific to one training run, we train v7s with the same recipe, data and exclusions but a narrower width (41.3 M vs 72.5 M parameters), a different seed and 60 k steps; its own step-10 k EMA serves as snapshot. Its CFG curve has the same shape (16 px: 19.29 / 19.10 / 20.34 / 22.96 / 28.00 for $w = 1.5 / 2 / 2.5 / 3 / 4$; 20 px: 61.62 / 57.48 / 56.30 / 59.73 / 63.32; 24 px: 99.23 / 93.78 / 93.96 / 94.72 / 99.47; seed 0), with best-CFG at $w = 2 / 2.5 / 2$ and a smaller over-guidance of the default (re-tuning alone −32 / −11 / −6 %). Table 3 shows the same ordering: the composed reference is −45 / −35 / −36 % below best-CFG at 16 / 20 / 24 px (−62 / −43 / −39 % below $w = 4$; seed 0), the label reference −33 / −19 / −17 % and autoguidance −27 / −30 / −29 %; the FD decomposition mirrors v7h (mean term 16.48 → 5.50 / 7.04 / 3.84 for label / autoguidance / composed; covariance 11.51 → 7.31 / 6.91 / 6.70; CFG at $w = 4$). Both models are our own and share data and lineage; the claim is replication within one family, not generality (§6).

**Table 3.** Second model v7s (clean), matched FD-DINOv2. 16 px: seed 0 / seed 1; 20 and 24 px: seed 0. Best CFG = lowest-FD CFG weight (in parentheses; seed 0). v7h row repeated from Table 1 (seed 0) for reference. Δ = composed vs best CFG (vs the $w = 4$ default in parentheses).

| Model | CFG $w=4$ (default) | Best CFG ($w$) | Label reference, $w=2$ | Autoguidance 10 k, $w=1.5$ | Composed, $w=1.5$ | Reverse `bucket:20`, $w=2$ | Δ composed vs best CFG (vs $w=4$) |
|---|---|---|---|---|---|---|---|
| v7h @16 (72.5 M) | 21.98 | 12.24 (1.5) | bk12 8.59 | 8.98 | **7.67** | 17.14 | −37 % (−65 %); 3-seed −40 % |
| v7s @16 (41.3 M) | 28.00 / 27.75 | 19.10 (2) | bk12 12.80 / 12.74 | 13.95 / 15.17 | **10.54 / 10.28** | 25.31 | −45 % / −46 % (−62 % / −63 %) |
| v7s @20 | 63.32 | 56.30 (2.5) | bk16 45.81 | 39.13 | **36.31** | — | −35 % (−43 %) |
| v7s @24 | 99.47 | 93.78 (2) | bk16 78.18 | 66.83 | **60.20** | — | −36 % (−39 %) |

### 4.6 Alignment–fidelity frontier

Guidance against a same-caption reference removes the unconditional CFG term, and the guided rows score lower CLIP similarity than CFG at the training default. Table 4 quantifies this with CLIP ViT-B/32 similarity (100·cos) and retrieval R@1 among 1 + 99 captions (chance 1 %) on the *same* saved samples: at 16 px every reference-guided row, autoguidance included, sits at 29.37–29.58 against 30.06 for CFG $w = 4$ (R@1 18 → 13–15 %); the gap grows to −0.9 at 32 px, is method-independent (snapshot and label references within 0.2), grows with $w$ (`bucket:12` $w = 3$: 29.25) and is absent for reverse references. CFG at $w = 4$ scores *above* the real sprites (30.06 vs 29.80) — it over-aligns — while the guided rows sit ≈ 0.3 below real.

This is not, however, a cost specific to reference guidance; it is the price of any guidance weight that reaches the FD optimum. Along the CFG weight curve (Figure 2; Appendix Table A2) CLIP rises monotonically with $w$ — 29.43 / 29.74 / 29.87 / 30.01 / 30.06 for $w = 1 / 1.5 / 2 / 3 / 4$ — while FD traces a U, so lowering CFG to its own FD optimum $w = 1.5$ costs exactly the CLIP the guided rows pay (29.74), and $w = 4$ buys its extra 0.32 CLIP with +9.7 FD. On the FD–CLIP plane every reference-guided row lies 3–5 FD below the best CFG point and further below the CFG curve at its own CLIP: `bucket:12` at (8.59, 29.37) against CFG $w = 1$ at (16.39, 29.43); autoguidance (8.98, 29.58) and composed (7.67, 29.51) against a CFG curve that passes at FD ≈ 14–15 for CLIP 29.5–29.6. PAG tracks the CFG curve (12.57–12.90 at CLIP 29.41–29.45). Two zero-training variants move along the frontier towards higher CLIP (Appendix Table A3): (a) `bucketu`, a reference under the lower bucket *and* the empty caption, so the guidance direction contains the CFG text direction at the same 2 NFE; (b) an additive plain-CFG term (`cfg_text`, 3 NFE). At 16 px the composed reference under `bucketu:12`, $w = 1.5$ — FD 8.60, CLIP 29.77, the real-data level (29.80; R@1 16.1 % vs 16.4 %) — and composed + `cfg_text` 1.5 (9.19, 29.72; 3 NFE) *dominate* the best CFG point (12.24, 29.74) in both coordinates. None of the seven variants reaches the $w = 4$ CLIP of 30.06; along the guided frontier each +0.1 CLIP costs about +0.5–1 FD, mostly in the mean term, against about +3 FD per 0.1 CLIP for CFG between $w = 1.5$ and 4.

At 20 / 24 / 32 px the both-coordinate dominance does not repeat verbatim: the best-CFG row keeps 0.54 / 0.56 / 0.65 CLIP over the composed row (30.02 vs 29.48, 29.89 vs 29.33, 29.43 vs 28.78), sitting above the real value at 20 / 24 px (+0.16 / +0.08) and below it at 32 px (−0.22). The alignment-recovering variant does at every resolution what it does at 16 px (Appendix Figure A1, seed 0): composed∘`bucketu:16` gives FD / CLIP 29.81 / 29.70 at 20 px (composed 29.66 / 29.48; best CFG 39.53 / 30.02) and 49.78 / 29.60 at 24 px (48.70 / 29.33; 67.34 / 29.89), and composed∘`bucketu:24` gives 65.27 / 29.18 at 32 px, better than the composed row in both coordinates (69.27 / 28.78; best CFG 83.65 / 29.43), with Inception FID / KID unchanged from composed (9.84 / 0.33, 13.24 / 1.08, 17.61 / 1.86). These rows still sit 0.16 / 0.21 / 0.47 CLIP below the real value. Against the CFG curve at *matched* CLIP, the CFG weight that reaches the same alignment ($w = 1.5$: 42.61 / 29.75, 71.22 / 29.70, 92.21 / 29.15) is 30 / 30 / 29 % worse in FD. At 12 px, `bucketu:16` composed with the snapshot gives 9.28 / 29.58 against best CFG 9.71 / 29.65 and autoguidance 8.54 / 29.41: dropping the text from the reference recovers alignment but a *higher*-bucket label still costs FD, so at 12 px we report autoguidance alone. The frontier statement holds at every resolution: at equal CLIP the guided rows are about 30 % below the CFG curve. The main tables keep the un-fixed composed row; the `bucketu` composed row is the operating point for real-level alignment, at +0.9 FD (about 3 seed sd) at 16 px.

**Figure 2.** Alignment–fidelity frontier at 16 px (`fig_pareto_fd_clip.png`; v7h, seed 0): FD-DINOv2 against CLIP 100·cos for the CFG weight curve ($w = 1 \ldots 10$), PAG ($w = 1.5 / 2 / 3$, and $w = 2$ + CFG term), `bucket:12`, autoguidance, composed, composed∘`bucketu:12`, composed + CFG term (1.5 / 2; 3 NFE) and the real held-out floor (dotted line: real CLIP 29.80). CFG cannot reach FD < 12.2 at any weight; the guided rows lie below the CFG curve at every CLIP level it reaches in the 29.4–29.9 range.

**Table 4.** CLIP ViT-B/32 100·cos (R@1 in parentheses) of the same samples as Tables 1–2; seed 0 / seed 1 where two values are given. "Best CFG" = the lowest-FD CFG weight of Table 2. Last column: composed reference under `bucketu:<lower>` (empty caption in the weak branch), $w = 1.5$, with its FD.

| $R$ | Real | CFG $w=4$ (default) | Best CFG | Lower-bucket ref, $w=2$ | Autoguidance, $w=1.5$ | Composed, $w=1.5$ | Higher-bucket ref | Composed∘`bucketu`, $w=1.5$ (FD / CLIP) |
|---|---|---|---|---|---|---|---|---|
| 12 | 29.55 (13.9 %) | 29.76 / 29.74 (15 %) | 29.65 (14.2 %) ($w=2$) | — | 29.41 / 29.39 (13 %) | — | bk16 29.32 (11.8 %) | bk16∘snap 9.28 / 29.58 |
| 16 | 29.80 (16.4 %) | 30.06 / 30.04 / 30.03 (18 %) | 29.74 (15.9 %) ($w=1.5$) | bk12 29.37 / 29.38 / 29.37 (13 %) | 29.58 / 29.53 / 29.57 (15 %) | 29.51 / 29.47 / 29.54 (14 %) | bk24 29.58; bk64 29.60 (15 %) | 8.60 / 29.77 (16.1 %) |
| 20 | 29.86 (19.3 %) | 30.16 / 30.09 (20 %) | 30.02 (18.3 %) ($w=2.5$) | bk16 29.34 / 29.35 (14 %) | 29.45 / 29.52 (15–16 %) | 29.48 / 29.49 (15 %) | bk32 29.62 (16 %) | 29.81 / 29.70 |
| 24 | 29.81 (20.2 %) | 30.11 / 30.12 (20 %) | 29.89 (19.3 %) ($w=2$) | bk16 29.09 / 29.11 (13 %) | 29.35 / 29.38 (15 %) | 29.33 / 29.34 (15 %) | bk32 29.48 (16 %) | 49.78 / 29.60 |
| 32 | 29.65 (20.9 %) | 29.72 (21.5 %) | 29.43 (20.6 %) ($w=2$) | bk24 28.49 (14 %) | 28.69 (15 %) | 28.78 (15 %) | bk48 28.85 (17.5 %) | 65.27 / 29.18 |

### 4.7 A second metric family and quantisation

Because DINOv2 sees one patch per pixel at these resolutions, we also compute Inception-v3 clean-FID and KID [TODO cite clean-fid] on the same saved samples (white composite, NEAREST ×4 to 64 px). Inception has far less headroom (floors 4.93 / 6.72 / 8.23 / 8.99 / 9.75 at 12–32 px; bare 9.57 vs floor 6.72 at 16 px, about 6.5× less than DINOv2). The composed reference's margin over best CFG is smaller under Inception — FID −8 / −10 / −14 / −11 % at 16 / 20 / 24 / 32 px (8.66 → 7.97, 10.66 → 9.62, 15.34 → 13.26, 19.37 → 17.24) against −40 / −27 / −29 / −17 % — but of the same sign at every resolution, and KID roughly halves (0.89 → 0.63, 0.67 → 0.26, 2.64 → 1.46, 2.77 → 1.81; Appendix Table A8). The Inception-optimal CFG weight at 16 px is $w = 2$ (8.66; $w = 1 / 1.5 / 3$: 10.65 / 9.31 / 8.86), not the DINOv2-optimal 1.5, so the 16 px margin is quoted against CFG's own best weight under that metric. PAG (mid, $w = 2$) scores 8.92 / 1.31, again indistinguishable from same-weight CFG. Two central claims survive the change of feature space (Table 5): the composed reference is best or tied-best at every resolution and on both models, and higher-bucket references are harmful; autoguidance and composed are within seed spread at 16 px, and the composed advantage is clear at 20 / 24 / 32 px. The *label-only* reference, however, is DINOv2-visible but Inception-weak: it lowers FID at 16 / 20 px but is flat or slightly worse at 24 / 32 px (16.13 → 16.45, KID 2.40 → 3.19) and lags autoguidance on v7s (9.77 vs 8.25). Rank agreement between the metric families over all saved rows is Spearman .98 / .86 / .96 / .68 / .83 at 12 / 16 / 20 / 24 / 32 px (Pearson .91–.99).

Quantisation tells the same story (Appendix A.9): 16-colour quantisation helps the bare $w = 4$ model substantially (21.98 → 12.64 / 45.92 → 42.65 / 79.80 → 64.38 / 96.85 → 81.41 at 16 / 20 / 24 / 32 px; best-CFG at 16 px only 12.24 → 11.55) and does not help, or hurts, the guided rows (composed 7.67 → 8.34 / 29.66 → 34.09 / 48.70 → 51.37 / 69.27 → 74.57). The composed row still wins at every resolution after quantisation, but its margin over $w = 4$ shrinks from −35 / −39 / −28 % to −20 / −20 / −8 % at 20 / 24 / 32 px. The label reference mainly corrects per-pixel colour and contrast statistics, which DINOv2 weights heavily and 64 px Inception barely sees; the snapshot corrects structure and coverage; the composition gets both.

**Table 5.** Inception-v3 clean-FID, same samples as Tables 1–3 (KID in Appendix Table A8). 16 px: seeds 0 / 1 / 2. Bare = CFG $w = 4$; Best CFG = the DINOv2-optimal weight of Table 2 re-scored under Inception (16 px also at the Inception-optimal $w = 2$).

| $R$ | Bare CFG ($w=4$) | Best CFG | Lower-bucket ref, $w=2$ | Autoguidance 10 k | Composed | Reverse (higher bucket) |
|---|---|---|---|---|---|---|
| 16 (3 seeds) | 9.57 / 9.61 / — | 9.31 ($w=1.5$); 8.66 ($w=2$) | bk12 8.61 / 8.63 / 8.64 | **7.84** / 7.97 / 7.71 | 7.97 / 7.86 / 7.95 | bk24 11.26; bk64 11.29 |
| 20 | 11.82 | 10.66 ($w=2.5$) | bk16 10.75 | 9.85 | **9.62** | bk32 15.45 |
| 24 | 16.13 | 15.34 ($w=2$) | bk16 16.45 (no gain) | 13.83 | **13.26** | bk32 22.56 |
| 32 | 19.77 | 19.37 ($w=2$) | bk24 20.10 (no gain) | 18.56 | **17.24** | bk48 25.83 |
| v7s @16 | 10.33 | — | bk12 9.77 | **8.25** | 8.49 | bk20 12.86 |
| v7s @20 / @24 | 13.06 / 16.82 | — | 11.73 / 17.85 | 10.42 / 14.66 | **10.22 / 13.85** | — |

## 5 What makes a reference effective: a characterisation and one intervention

Why does a lower bucket label give a useful weak reference while a higher one does not, and what does the correction change? This section is a characterisation of the pure weak-reference beliefs, an FD decomposition, two trained reference branches and one untrained interventional control; it constrains what the reference must contain but does not prove a mechanism. "Strong model" statistics are of the CFG $w = 4$ samples; the statistics of the best-CFG ($w = 1.5$) samples were not measured [TBD]. All statistics are seed 0.

### 5.1 What the lower-resolution belief looks like

We sample each candidate reference alone (`--cfg 0`, following $e_{\text{ref}}$ only; 3000 samples, matched protocol) and measure over opaque pixels the median number of unique colours per sprite (ncol), the fraction of exactly equal adjacent pixel pairs (flat) and the mean adjacent $|\Delta\mathrm{RGB}|$ (TV). Table 6 gives the 16 px statistics. The bucket:12 belief is *not* a simpler image in the sense of real low-resolution sprites (real 12 px sprites: median 5 colours, flat ≈ .51): it has *more* colours than the strong prediction (78 vs 73) and essentially no flat regions (.015), which rules out a "simplicity prior". What distinguishes it is local contrast, TV 21.6 against 32.0 — a soft, low-contrast rendering of the same sprite (Figure 1, eighth row). The label acts as a monotone contrast knob on the same network (TV 21.6 → 31.9 → 40.9 for bucket 12 → 24 → 64), and only the low-contrast end is a useful reference. The pattern replicates at 20 and 24 px (Appendix Table A7): the references that help have TV well below the strong model (bucket:16 21.8 at 20 px, 18.2 at 24 px; bucket:12 16.9 at 20 px); those that do not have TV at or above it (bucket:24 28.5 at 20 px, guided FD ≈ bare; bucket:32 31.3 at 24 px, guided FD 101.67 vs bare 79.80). The reference's *own* quality is anti-correlated with its usefulness: at 20 px bucket:24 is the best-looking belief (FD 66.07 vs 94.02 and 179.63) and the worst reference; at 24 px bucket:32 is the best-looking (60.16) and actively harmful.

**Table 6.** Statistics of the *pure* weak-reference samples at 16 px (sampled with `--cfg 0`) and the FD obtained when each is used as the guidance reference; v7h, seed 0. Rows marked ‡ are trained branches (§5.4) on fine-tuned weights; for `probe_cg` the statistics are those of its training target (block-averaged real sprites), not of the sampled branch, which was not measured.

| Reference (sampled alone) | FD alone | ncol | flat | TV | FD when used as reference |
|---|---|---|---|---|---|
| Real 16 px sprites | 3.45 | 34 | .202 | 30.2 | — |
| Strong model (v7h, CFG $w=4$) | 21.98 | 73 | .030 | 32.0 | — |
| **bucket:12 belief** (same weights) | 36.43 | 78 | .015 | **21.6** | **8.59** ($w=2$) |
| Snapshot 10 k | 69.16 | 95 | .000 | 27.3 | 8.98 ($w=1.5$) |
| Unconditional (no text) | 32.15 | 68 | .017 | 27.3 | 12.24 (CFG, best $w=1.5$); 21.98 ($w=4$) |
| bucket:24 belief | 34.50 | 75 | .039 | 31.9 | 17.27 ($w=2$) |
| bucket:64 belief | 327.6 | 102 | .046 | 40.9 | 19.34 ($w=2$) |
| ‡ `probe_cg` block-average branch (training-target stats) | — | 21 | .605 | 14.2 | 21.53 / 35.22 / 59.38 ($w=1.5/2/3$; bare 19.17) |
| ‡ `probe_cc` contrast-shrunk branch | 41.37 | 64 | .034 | 15.9 | 14.71 / 19.98 / 44.90 ($w=1.5/2/3$; bare 20.63) |
| *After guidance*: autoguidance / bucket:12 / composed | 8.98 / 8.59 / 7.67 | 64 / 61 / 60.5 | .041 / .045 / .044 | 29.0 / 32.4 / 30.1 | — |

### 5.2 TV predicts the guided FD, with one instructive exception

Figure 3 plots pure-reference TV against the FD obtained when that reference guides, for the 16 px references of Table 6. Normalising per resolution — $x = \mathrm{TV}_{\text{ref}}/\mathrm{TV}_{\text{strong}}$, $y = \mathrm{FD}_{\text{guided}}/\mathrm{FD}_{\text{bare}}$ — for the ten bucket and snapshot references at 16 / 20 / 24 px (Appendix Figure A6), every same-caption belief with $x < 1$ gives $y < 1$, and every reference with $x \ge 1$ gives $y \approx 1$ or $y > 1$. This is a characterisation, not yet a mechanism: TV is a statistic of the pure belief that co-varies with usefulness across hand-chosen references, and "necessary" is inferred from two failures per resolution.

The single violation is the unconditional prediction: TV 27.3, as low as the snapshot, yet far less gain (12.24 at the best CFG weight against 8.98). It clarifies the rule: the unconditional branch is not a same-condition belief, and its difference from the strong prediction carries the *text* direction. Low TV is a property of the pure belief, but guidance extrapolates along the *difference*, which is a contrast correction only when the two predictions agree on everything but contrast. The same data support the structure-alignment clause: at 20 px the farthest lower bucket (12) has the lowest TV (16.9) but drifts in opacity (.455 vs .362 for the strong model) and is the weaker reference (35.85 vs 32.89); at 32 px the farther bucket is clearly worse (91.26 vs 77.45); the one counter-example is 24 px, where bucket:12 (57.70) marginally beats bucket:16 (60.41) at seed 0. Lower TV is better until the belief starts to drift structurally.

**Figure 3.** TV of the pure weak reference against FD-DINOv2 after guidance, 16 px (`fig_tv_vs_fd.png`). Circles: the model's own beliefs (bucket:12, snapshot 10 k, unconditional via CFG $w = 4$, bucket:24, bucket:64); crosses: the trained degraded-view branches of §5.4 (block-average at $w = 1.5$ and 2, contrast-shrunk at $w = 1.5$). Dashed: strong-model TV 32.0; dotted: bare CFG $w = 4$, 21.98. Every same-caption belief left of the dashed line helps; the trained branches with the lowest TV do not.

### 5.3 What each reference corrects

*FD decomposition.* Splitting FD into mean and covariance terms separates the two references (Appendix Table A6, 16 px seed 0). Along the CFG curve the mean term is never below 5.87 (9.47 / 5.87 / 6.32 / 8.70 / 12.13 for $w = 1 / 1.5 / 2 / 3 / 4$): re-tuning halves the mean shift but no weight removes it. The label reference has the lowest mean term (2.55) but a weaker covariance term and coverage (6.04, .892); snapshot autoguidance a larger mean term (3.88) but the best covariance term and coverage (5.09, .928); the composed reference keeps the label's mean term (2.40) and most of the snapshot's covariance gain (5.27; coverage .909). v7s reproduces the split (§4.5). The label removes a systematic shift of the generated distribution — the per-pixel colour and contrast error — that the CFG weight cannot reach; the snapshot restores coverage. Higher buckets do the opposite: their belief has contrast at or above the strong prediction, so $e_{\text{strong}} - e_{\text{ref}}$ points towards *lower* contrast, with the over-guidance signature of §4.4 (precision ↑ .935, recall ↓ .85–.88; Appendix A.14).

*Timestep localisation.* Interval scheduling (Appendix Table A10) localises the snapshot in time: snapshot weights only for $t/T \in [0, 0.5]$ give 7.70 (mean / cov 2.42 / 5.28), indistinguishable from the full composed rule (7.67, 2.5 / 5.2); only for $[0.5, 1]$, 8.99 (3.54 / 5.45), close to the label reference alone (8.59); $[0.2, 0.8]$, 8.84. The snapshot's information lives in the low-noise half; at high noise the 10 k snapshot and the final weights share the same layout belief, while the label reference acts throughout.

*Geometry of the correction.* Components that fail on top of the composed reference (Appendix Table A10) fix its geometry: APG [cite: Sadat2024apg], which removes the component of the guidance difference parallel to $x_0$, raises the composed FD from 7.67 to 9.94; FDG [cite: Sabour2025] with $w_{\text{low}} < w_{\text{high}}$ raises the mean term monotonically (2.5 → 3.54 → 4.8 for $w_{\text{low}} = 1.5, 1.25, 1.0$); with channel-decoupled weights FD is set almost entirely by the RGB weight (1.5 → 8.35, 2 → 11.16, 2.5 → 19.17, 3 → 32.36). The useful correction is radial, low-frequency and in the colour channels: per-pixel contrast, not high-frequency detail.

### 5.4 Controlled references: low TV is necessary but not sufficient, and the correction is not a scalar

If the reading is "extrapolate away from a low-contrast, structure-aligned belief", a trained branch built to have those properties should work, one with low contrast that imposes its own structure should not, and — the sharpest test — lowering the contrast of the strong prediction itself should reproduce the label's effect if the label were nothing more than a contrast knob. We run all three tests; the first two are trained, the third is an untrained intervention. The trained probes fine-tune v7h with a second label set, replacing the training view by a degraded view under the second label [TODO cite log for the probe recipe: 20 k steps, class embedding 7 → 14, replacement probability 0.5]; each probe is compared with the label reference and snapshot autoguidance *on the same fine-tuned weights* (Appendix Table A5), so fine-tuning is not a confound.

*Block average (`probe_cg`).* Degraded view = 2 × 2 block average, nearest-neighbour upsampled. Its training target has TV 14.2 but flat .605 and ncol 21 — a block grid the strong prediction lacks. As reference it is *harmful*: 21.53 / 35.22 / 59.38 for $w = 1.5 / 2 / 3$ against 19.17 bare on the same weights, while the paired label reference (10.12) and snapshot (9.10) work as usual.

*Contrast shrink (`probe_cc`).* Degraded view = RGB shrunk 0.6× towards the per-image opaque mean: structure-aligned, no grid. Sampled alone: FD 41.37, ncol 64, flat .034, TV 15.9. As reference it is *effective* — 14.71 at $w = 1.5$ vs 20.63 bare on the same weights — so replacing the block grid by a structure-aligned degradation flips the sign of the effect. But it is far weaker than the free label on the same weights (9.43) and the paired snapshot (8.77): its TV 15.9 is below the bucket:12 belief's 21.6, yet ncol falls to 64 and flat rises to .034, a "fewer colours, flatter" bias that extrapolation amplifies in reverse. It also fails to fill the 12 px gap: 9.66 ($w = 1.25$) vs 8.16 for snapshot autoguidance and 11.78 bare on the same weights.

*Uniform contrast shrink of the strong prediction itself (`shrink:f`; interventional, untrained, 1 NFE).* At every step we take the strong model's own $\hat{x}_0$, shrink its RGB contrast towards the per-image opaque mean by a factor $f$, and use the result as the weak reference. The reference is then perfectly structure-aligned and lower in contrast by construction, and the guidance update reduces to a uniform amplification of the strong prediction's contrast about its mean; if the lower-bucket belief were "the strong prediction with less contrast", this would reproduce its gain. It does the opposite, monotonically: $f = 0.85 / 0.7 / 0.5$ at $w = 2$ give FD 74.8 / 200.4 / 374.6, and $f = 0.7$ at $w = 3$ gives 444.4 (CLIP 28.5–29.5, R@1 5–12 %), against 12.24 for best-CFG, 8.59 for the bucket:12 reference and 16.39 for no guidance at all. The lower-bucket belief's difference from the strong prediction is therefore *spatially structured* — a per-pixel pattern of where and by how much contrast is missing — not a global scalar; the trained per-pixel shrink sits between the two, recovering only a weak version of the direction the label supplies for free.

Three references with TV 14.2 / 15.9 / 21.6 thus span harmful / weakly effective / effective, in reverse order of TV, and the scalar shrink — the lowest-effort way to obtain "lower TV, same structure" — is the most harmful of all. Lower TV than the strong prediction is necessary (bucket:24/64 fail) but not sufficient: the effective reference must differ from the strong prediction in a spatially structured way, with colour count, opacity and layout otherwise matched, which the label satisfies because it changes the belief and nothing else, and which no hand-designed degradation we tried does.

### 5.5 What remains

After guidance the sample statistics move towards the real distribution without reaching it (Table 6, last row): ncol 73 → 61 / 64 / 60.5 (label / snapshot / composed) against 34 real; flat .030 → .045 / .041 / .044 against .202. Guided samples still have about 1.8× the colour count and a quarter of the flat-region fraction of real sprites, and the remaining gap (7.53 vs the 3.45 floor) has the form of residual colour mixing. The label reference alone does not lower the TV of the *guided* samples (32.4 vs 32.0 at 16 px), so "restoring contrast" means extrapolating away from a low-contrast belief, not a monotone change of the TV statistic.

## 6 Limitations and Conclusion

**Text alignment.** The guided rows sit at the CLIP level of CFG at $w \approx 1$–1.5, 0.3 below the real value at 16 px and up to 0.9 below the $w = 4$ row at 32 px (Table 4); the cost is shared by autoguidance and is the price CFG itself pays for lowering $w$ to its FD optimum (§4.6). The `bucketu` variant recovers real-level CLIP at 16 px at +0.9 FD, but at 20–32 px the guided rows remain 0.16–0.47 CLIP below the real value, no variant reaches the $w = 4$ CLIP of 30.06, and no human study distinguishes "aligned" from "over-aligned".

**The label-only gain shrinks with resolution and is partly metric-specific.** The free variant matches autoguidance at 16 px but gains only −17 / −12 / −7 % at 20 / 24 / 32 px, is Inception-flat or slightly worse at 24 / 32 px, lags autoguidance on v7s under Inception, and loses most of its advantage under 16-colour quantisation; after q16 autoguidance beats the composed row at 16 px (7.8 vs 8.3–8.4). The composed reference and the reverse-control result hold under every metric and against best-CFG at every resolution, but the composed reference stores a snapshot. Inception margins are smaller (−8 to −14 %) than DINOv2 margins; with about 6.5× less headroom its relative gains are compressed.

**Coverage, tuning and scope.** The method needs a lower bucket; at 12 px only autoguidance applies (9.71 → 8.54 ± 0.73), and a trained contrast-shrunk label does not fill the gap. The CFG weight, our fixed $w$ and the 24 px bucket choice were selected on the held-out FD we report — there is no separate validation split — and most rows beyond the 16 / 20 / 24 px headlines are seed 0. Both models are our own, trained on one dataset (OGA-derived sprites, noisy BLIP captions) within one architecture family; at 20–32 px the bare model is 3–7× above the floor, so the gain is largest where the model is weakest. We have not tested a public bucketed model (SDXL size conditioning, Matryoshka, FiT), so the relation to `negative_original_size` remains an analogy, and there is no human preference study. Section 5 is a characterisation plus one intervention, not a proven mechanism.

**Conclusion.** A bucketed multi-resolution diffusion model contains its own compatible weak model: the same weights under a lower-resolution label. Used as a guidance reference it replaces CFG at CFG's cost, matches snapshot autoguidance at 16 px, and composed with an early snapshot lowers FD-DINOv2 by 17–45 % below the best CFG weight at every resolution and on both models tested, at the CLIP level of CFG's own optimum. The effect is directional, the reference's usefulness tracks its total variation relative to the strong prediction, and a scalar contrast shrink cannot reproduce it. Whether the same free reference exists in large bucketed models is the natural next test.

## Appendix

Contents (everything demoted from the main body): A.1 CFG weight curves and zero-training baselines (Table A1); A.2 the 16 px frontier table (Table A2); A.3 alignment variants and the frontier at 12–32 px (Table A3, Figure A1); A.4 qualitative figures at 12 / 20 / 24 / 32 px (Figures A2–A5); A.5 reverse controls (Table A4); A.6 trained degraded-view branches, paired controls (Table A5); A.7 FD decomposition (Table A6); A.8 pure-belief statistics at 20 / 24 px (Table A7, Figure A6); A.9 16-colour quantisation at 20 / 24 / 32 px; A.10 Inception clean-FID / KID in full (Table A8); A.11 cost (Table A9); A.12 sampler robustness; A.13 falsified second components (Table A10); A.14 why higher-bucket references fail; A.15 an operational rule for reference selection; A.16 the contaminated third model; A.17 broader impact.

### A.1 CFG weight curves, guidance-weight sweeps and other zero-training baselines

Beyond the three facts summarised in §4.3: the label reference has a flat weight curve at every resolution tested — every weight in $w \in \{1.25, \ldots, 3\}$ at 20 / 24 px beats the $w = 4$ default (45.92 / 79.80, seed 0), and every weight in $[1.5, 3]$ at 20 px and $[1.5, 2.5]$ at 24 px beats best-CFG (39.53 / 67.34, seed 0) — whereas the snapshot reference has a sharp optimum and explodes past it (at $w \ge 2.5$ it is worse than best-CFG at 20 px). At 24 px the snapshot optimum moves to $w = 2$ (52.52 vs 53.36), so the composed row at $w = 1.5$ is not tuned in the snapshot's favour. Autoguidance with 200 DDPM steps at $w = 2$ gives 11.35 (100 steps: 11.33); combining autoguidance $w = 2$ with a CFG $w = 7$ interval gives 133.35.

**Table A1.** Weight sweeps and zero-training baselines, v7h, seed 0 unless a ± is given (3 seeds). 16 px unless stated; best-CFG values in bold in the CFG rows. 20 / 24 px reference sweeps against best-CFG 39.53 / 67.34 (seed 0; 39.57 ± 0.04 / 67.77 ± 0.77 over three seeds) and the $w = 4$ default 45.92 / 79.80 (seed 0; 47.04 ± 1.11 / 78.96 ± 0.74).

| Reference | $w$ sweep | FD |
|---|---|---|
| CFG (unconditional), 16 px | 1 / 1.5 / 2 / 2.5 / 3 / 4 (default) / 7 / 10 | 16.39 / **12.24** / 12.98 / 14.56 / 16.67 / 21.98 / 42.95 / 62.12 |
| CFG $w=1.5$ (best), 16 px, seeds 0 / 1 / 2 | 1.5 | 12.24 / 13.32 / 11.91 = **12.49 ± 0.72** |
| CFG, 12 px | 1.5 / 2 / 2.5 / 3 / 4 | 10.44 / **9.71** / 10.53 / 11.09 / 13.02 ± 0.28 |
| CFG, 20 px | 1.5 / 2 / 2.5 / 3 / 4 | 42.61 / 40.57 / **39.53** / 42.99 / 45.92 (47.04 ± 1.11) |
| CFG $w=2.5$ (best), 20 px, seeds 0 / 1 / 2 | 2.5 | 39.53 / 39.57 / 39.61 = **39.57 ± 0.04** |
| CFG, 24 px | 1.5 / 2 / 3 / 4 | 71.22 / **67.34** / 72.93 / 79.80 (78.96 ± 0.74) |
| CFG $w=2$ (best), 24 px, seeds 0 / 1 / 2 | 2 | 67.34 / 68.66 / 67.31 = **67.77 ± 0.77** |
| CFG, 32 px | 1.5 / 2 / 3 / 4 | 92.21 / **83.65** / 86.99 / 96.85 |
| CFG, v7s @16 px | 1.5 / 2 / 2.5 / 3 / 4 | 19.29 / **19.10** / 20.34 / 22.96 / 28.00 |
| CFG, v7s @20 px | 1.5 / 2 / 2.5 / 3 / 4 | 61.62 / 57.48 / **56.30** / 59.73 / 63.32 |
| CFG, v7s @24 px | 1.5 / 2 / 2.5 / 3 / 4 | 99.23 / **93.78** / 93.96 / 94.72 / 99.47 |
| CFG + CADS, 16 px | $w=4$ / $w=7$ / $w=7$, $s=.25$ | 44.58 / 71.17 / 88.61 |
| CFG $w=7$, interval $[0,.8]$, 16 px | — | 32.57 |
| PAG `pag:mid` [cite: Ahn2024], 16 px | 1.5 / 2 / 3 | 12.57 / 12.62 / 12.90 |
| PAG mid + `down_blocks.1`, 16 px | 2 | 13.05 |
| PAG `pag:mid` $w=2$ + CFG term 1.5 (3 NFE), 16 px | — | 11.73 |
| Autoguidance (snapshot 10 k), 16 px | 1.5 / 2 / 2.5 / 3 | 8.98 / 11.33 / 19.19 / 30.11 |
| Autoguidance, $w=2$, snapshot step | 5 k / 10 k / 20 k / 40 k | 14.89 / 11.33 / 12.14 / 11.62 |
| Label reference `bucket:12`, 16 px | 1.5 / 2 / 2.5 / 3 | 10.21 / 8.59 / 11.20 / 13.87 |
| `bucketmix:12` (½ lower bucket + ½ unconditional), 16 px | 2 | 9.42 |
| Composed (10 k under `bucket:12`), 16 px | 1.25 (seed 1) / 1.5 / 2 | 8.98 / 7.67 / 10.83 |
| Composed, other snapshots, $w=1.5$ | 20 k / 5 k (seed 1) | 8.03 / 8.68 |
| Autoguidance, channel-decoupled $w$ (RGB / alpha) | 2/1 / 2/3 / 3/2 / 1.5/2.5 / 2.5/1.5 | 11.16 / 10.74 / 32.36 / 8.35 / 19.17 |
| Degraded-input reference: 2×2 box-blurred $x_t$ / 1-px cyclic shift | 2 and 1.5 / — | 223.07 and 99.89 / 18.95 |
| Label reference `bucket:16`, 20 px | 1.25 / 1.5 / 2 / 2.5 / 3 | 40.96 / 37.48 / **32.89** / 35.15 / 38.53 |
| Autoguidance (snapshot 10 k), 20 px | 1.25 / 1.5 / 2 / 2.5 / 3 | 36.75 / **31.78** / 33.66 / 42.02 / 59.04 |
| Label reference `bucket:16`, 24 px | 1.25 / 1.5 / 2 / 2.5 / 3 | 71.12 / 63.09 / **60.41** / 65.77 / 74.61 |
| Autoguidance (snapshot 10 k), 24 px | 1.25 / 1.5 / 2 / 2.5 / 3 | 64.85 / 53.36 / **52.52** / 66.53 / 83.41 |

### A.2 The CFG weight curve and the 16 px alignment–fidelity frontier

**Table A2.** The CFG weight curve and the alignment–fidelity frontier, 16 px, v7h, seed 0 (plotted in Figure 2). FD with mean / covariance terms, CLIP 100·cos and R@1 on the same saved samples.

| Row | NFE | FD | mean / cov | CLIP | R@1 |
|---|---|---|---|---|---|
| CFG $w=1$ | 2 | 16.39 | 9.47 / 6.92 | 29.43 | 13.8 % |
| CFG $w=1.5$ (**best CFG**) | 2 | **12.24** | 5.87 / 6.37 | 29.74 | 15.9 % |
| CFG $w=2$ | 2 | 12.98 | 6.32 / 6.66 | 29.87 | 16.8 % |
| CFG $w=2.5$ | 2 | 14.56 | — | 29.95 | 17.1 % |
| CFG $w=3$ | 2 | 16.67 | 8.70 / 7.97 | 30.01 | 17.4 % |
| CFG $w=4$ (training default) | 2 | 21.98 | 12.13 / 9.85 | 30.06 | 18.3 % |
| CFG $w=7$ | 2 | 42.95 | 27.01 / 15.94 | 30.04 | 17.9 % |
| CFG $w=10$ | 2 | 62.12 | 41.15 / 20.97 | 30.05 | 17.3 % |
| PAG `pag:mid` $w = 1.5$ / 2 / 3 | 2 | 12.57 / 12.62 / 12.90 | — | 29.45 / 29.41 / 29.45 | — |
| PAG `pag:mid` $w=2$ + CFG term 1.5 | 3 | 11.73 | — | 29.74 | — |
| `bucket:12` $w=2$ (ours) | 2 | **8.59** | 2.55 / 6.04 | 29.37 | 13.1 % |
| Autoguidance $w=1.5$ | 2 | 8.98 | 3.88 / 5.09 | 29.58 | 15 % |
| Composed $w=1.5$ (ours) | 2 | **7.67** | 2.40 / 5.27 | 29.51 | 14.4 % |
| Composed under `bucketu:12`, $w=1.5$ (ours) | 2 | 8.60 | 2.38 / 6.21 | **29.77** (real: 29.80) | 16.1 % |
| Composed $w=1.5$ + `cfg_text` 1.5 | 3 | 9.19 | 3.19 / 5.99 | 29.72 | 15.4 % |
| Composed $w=1.5$ + `cfg_text` 2 | 3 | 11.10 | 4.26 / 6.85 | 29.89 | 15.9 % |

### A.3 Alignment variants and the frontier at 12–32 px

Two zero-training variants move along the frontier towards higher CLIP: (a) `bucketu:12` = reference under the lower bucket *and* the empty caption, so the guidance direction contains the CFG text direction (same 2 NFE); (b) `cfg_text` = an additive plain-CFG term on top of the reference term (3 NFE). None of the seven rows reaches CLIP ≥ 30.06; the points lie on an FD–CLIP frontier (CLIP 29.37 @ 8.59 → 29.77 @ 8.60 / 10.45 → 29.89 @ 11.10), every +0.1 CLIP costs ≈ +0.5–1 FD, and the FD lost is mostly mean term (the colour/contrast shift returns as the CFG direction re-enters). The cheapest point is the composed reference under `bucketu:12` at $w = 1.5$: FD 8.60 (+0.9 over composed, about 3 seed sd), CLIP back to the real-data level (29.77 vs 29.80; R@1 16.1 % vs 16.4 %), same 2 NFE. The `bucketu` rows are also the analogue of SDXL's `negative_original_size` practice in our model (§2): alone, `bucketu:12` gives 10.45 ($w = 1.5$) / 12.10 ($w = 2$), 1.9–3.5 FD worse than the same-caption label reference (8.59).

**Table A3.** Alignment variants along the frontier, 16 px, seed 0.

| Row | NFE | FD | +q16 | mean / cov | CLIP | R@1 |
|---|---|---|---|---|---|---|
| CFG $w=4$ (training default) | 2 | 21.98 | 12.64 | 12.13 / 9.85 | 30.06 | 18.3 % |
| CFG $w=1.5$ (best CFG) | 2 | 12.24 | 11.55 | 5.87 / 6.37 | 29.74 | 15.9 % |
| `bucket:12` $w=2$ | 2 | 8.59 | 8.99 | 2.55 / 6.04 | 29.37 | 13.1 % |
| Composed $w=1.5$ | 2 | **7.67** | 8.34 | 2.40 / 5.27 | 29.51 | 14.4 % |
| (a) `bucketu:12` $w=2$ | 2 | 12.10 | 9.24 | 5.15 / 6.96 | 29.84 | 16.1 % |
| (a) `bucketu:12` $w=1.5$ | 2 | 10.45 | 8.60 | 4.48 / 5.97 | 29.77 | 16.4 % |
| (a) composed under `bucketu:12`, $w=1.5$ | 2 | 8.60 | 9.32 | 2.38 / 6.21 | 29.77 | 16.1 % |
| (a) composed under `bucketu:12`, $w=1.25$ | 2 | 9.46 | 9.60 | 3.54 / 5.91 | 29.65 | 15.7 % |
| (b) `bucket:12` $w=2$ + `cfg_text` 1.5 | 3 | 9.50 | 8.71 | 3.25 / 6.24 | 29.71 | 16.2 % |
| (b) composed $w=1.5$ + `cfg_text` 1.5 | 3 | 9.19 | 8.57 | 3.19 / 5.99 | 29.72 | 15.4 % |
| (b) composed $w=1.5$ + `cfg_text` 2 | 3 | 11.10 | 9.48 | 4.26 / 6.85 | 29.89 | 15.9 % |

**Figure A1.** Alignment–fidelity frontier at 12 / 20 / 24 / 32 px (`fig_pareto_allR.png`, four panels; v7h, seed 0): CFG weight curve, best-CFG point, lower-bucket reference, autoguidance, composed and composed∘`bucketu:<lower>` (at 12 px: `bucketu:16`∘snapshot), with the real held-out CLIP marked. Other CFG points: @20 $w = 2$ CLIP 29.92; @24 $w = 3$ 30.00; @32 $w = 3$ 29.60; @12 $w = 1.5$ / 3 29.54 / 29.76. At matched CLIP (CFG $w = 1.5$) the composed∘`bucketu` rows are 30 / 30 / 29 % lower in FD at 20 / 24 / 32 px.

### A.4 Qualitative samples at other resolutions

**Figures A2–A5.** Prompt-aligned samples at 12, 20, 24 and 32 px (`fig_qual_12px.png`, `fig_qual_20px.png`, `fig_qual_24px.png`, `fig_qual_32px.png`; same seed and caption per column; FD in parentheses, seed 0). **A2, 12 px** (no lower bucket, so no label or composed rows): real; CFG $w = 4$ (12.91); best-CFG $w = 2$ (9.71); autoguidance 10 k, $w = 1.5$ (9.18); reverse `bucket:16` reference, $w = 2$ (14.45). **A3, 20 px**: real; CFG $w = 4$ (45.92); best-CFG $w = 2.5$ (39.53); autoguidance (31.78); `bucket:16` self-guidance, $w = 2$ (32.89); composed (29.66); composed∘`bucketu:16` (29.81); the `bucket:16` belief alone (94.02); reverse `bucket:32`, $w = 2$ (59.13). **A4, 24 px**: real; CFG $w = 4$ (79.80); best-CFG $w = 2$ (67.34); autoguidance (53.36); `bucket:16` self-guidance, $w = 2$ (60.41); composed (48.70); composed∘`bucketu:16` (49.78); the `bucket:16` belief alone (173.39); reverse `bucket:32`, $w = 2$ (101.67). **A5, 32 px**: real; CFG $w = 4$ (96.85); best-CFG $w = 2$ (83.65); autoguidance (75.23); `bucket:24` self-guidance, $w = 2$ (77.45); composed (69.27); composed∘`bucketu:24` (65.27); reverse `bucket:48`, $w = 2$ (113.93).

### A.5 Reverse controls in full

**Table A4.** Reverse controls (higher bucket as weak reference, $w = 2$), v7h, seed 0. Effects are given against the $w = 4$ default and, after the slash, against best-CFG.

| $R$ | CFG $w=4$ (default) | Best CFG | Best lower / composed | Higher-bucket reference | Effect vs $w=4$ / vs best CFG |
|---|---|---|---|---|---|
| 12 | 12.91 | 9.71 | — / autoguidance 9.18 | bk16 **14.45** | worse (+1.5) / worse (+4.7) |
| 16 | 21.98 | 12.24 | bk12 8.59 / 7.67 | bk20 17.14; bk24 17.27; bk64 19.34 | slightly better raw, worse after q16 (18.64 / 21.02 / 24.56 vs 12.64) / worse (+4.9 … +7.1) |
| 20 | 45.92 | 39.53 | bk16 32.89 / 29.66 | bk24 **46.62**; bk32 **59.13** | ≈ bare (+0.7) / worse (+13.2) ; worse (+7.1) / worse (+19.6) |
| 24 | 79.80 | 67.34 | bk16 60.41 / 48.70 | bk32 **101.67** | worse (+21.9) / worse (+34.3) |
| 32 | 96.85 | 83.65 | bk24 77.45 / 69.27 | bk48 **113.93** | worse (+17.1) / worse (+30.3) |

### A.6 Trained degraded-view branches: same-weights paired controls

The trained probes fine-tune v7h with a second label set [TODO cite log: 20 k steps, class embedding 7 → 14, replacement probability 0.5] and sample with $e = e_{\text{coarse}} + w\,(e_{\text{fine}} - e_{\text{coarse}})$. The bare column is CFG at $w = 4$; no CFG weight sweep was run on the fine-tuned weights, so these rows are compared within the table only. Mean-term values quoted in an earlier draft for `probe_cg` (13.32 → 42.05, bare 9.95) and `probe_cc` (8.1 vs 10.97 vs 3.25) have no source in the outline and are omitted pending a log citation.

**Table A5.** Same-weights paired controls inside the trained probes, 16 px, seed 0 (12 px row: no lower bucket); last row: the untrained `shrink:f` control on v7h.

| Fine-tuned weights | Bare CFG $w=4$ | Trained branch, $w=1.5$ | `bucket:12`, $w=2$ | Autoguidance 10 k, $w=1.5$ |
|---|---|---|---|---|
| `probe_cg` (2×2 block average) | 19.17 (q16 12.37) | 21.53 (q16 16.05) | 10.12 (q16 9.51) | 9.10 (q16 8.45) |
| `probe_cc` (contrast shrink 0.6×) | 20.63 (q16 12.66) | **14.71** (q16 9.17) | 9.43 (q16 8.97) | 8.77 (q16 7.78) |
| `probe_cc` @12 px | 11.78 | 9.66 ($w=1.25$) / 10.07 ($w=1.5$) | — | 8.16 |
| v7h, `shrink:f` (untrained, 1 NFE): $f = 0.85 / 0.7 / 0.5$, $w=2$; $f=0.7$, $w=3$ | best CFG 12.24 ($w=4$: 21.98) | **74.8 / 200.4 / 374.6; 444.4** | 8.59 | 8.98 |

### A.7 FD decomposition and precision / recall at 16 px

**Table A6.** FD decomposition (mean and covariance terms) and precision / recall / density / coverage for the seed-0 rows of Table 1 (composed: seed 0 / seed 1).

| Setting | FD | mean term | cov term | precision | recall | density | coverage |
|---|---|---|---|---|---|---|---|
| Real held-out (floor) | 3.45 | 0.38 | 3.07 | .939 | .924 | 1.029 | .966 |
| CFG $w=4$ | 21.98 | 12.13 | 9.85 | .907 | .902 | .874 | .851 |
| CFG $w=1.5$ (best CFG) | 12.24 | 5.87 | 6.37 | — | — | — | — |
| Autoguidance 10 k, $w=1.5$ | 8.98 | 3.88 | 5.09 | .913 | .906 | .936 | .928 |
| Autoguidance 10 k, $w=2$ | 11.33 | 5.09 | 6.24 | .915 | .902 | .895 | .894 |
| `bucket:12`, $w=2$ | 8.59 | 2.55 | 6.04 | .911 | .894 | .865 | .892 |
| Composed 10 k + `bucket:12`, $w=1.5$ (seed 0 / seed 1) | 7.67 / 7.32 | 2.40 / 2.09 | 5.27 / 5.23 | [TBD] | [TBD] / .902 | [TBD] | .909 / .916 |

The bucket reference has the lowest mean term (systematic mean shift), the snapshot the best coverage; the composition takes roughly half of each reference's benefit. On v7s: mean term 16.48 → 5.50 (bucket:12) / 7.04 (autoguidance) / 3.84 (composed); cov term 11.51 → 7.31 / 6.91 / 6.70.

### A.8 Pure-belief statistics at 20 and 24 px

**Table A7.** Statistics of the pure weak-reference samples (sampled with `--cfg 0`, matched protocol, seed 0) at 20 and 24 px, with the opaque-pixel fraction, and the FD obtained when each is used as reference ($w = 2$).

| $R$ | Reference (sampled alone) | FD alone | opaque | ncol | flat | TV | FD as reference ($w=2$) | Verdict |
|---|---|---|---|---|---|---|---|---|
| 20 | Real 20 px | 12.14 | .345 | 33 | .277 | 29.9 | — | — |
| 20 | Strong, v7h CFG $w=4$ | 45.92 | .362 | 96 | .056 | **31.0** | — | — |
| 20 | **bucket:16 belief** | 94.02 | .395 | 107 | .030 | **21.8** | **32.89** | effective |
| 20 | bucket:12 belief | 179.63 | .455 | 121 | .018 | **16.9** | 35.85 | effective, weaker (opaque .455 ≫ .362: structure drift) |
| 20 | bucket:24 belief | 66.07 | .355 | 97 | .038 | 28.5 (≈ strong) | 46.62 (≈ bare) | ineffective |
| 20 | *after guidance*: bk16 $w$2 / composed | 32.89 / 29.66 | .339 / .350 | 81 / 78 | .072 / .072 | 32.5 / 29.7 | — | — |
| 24 | Real 24 px | 12.96 | .333 | 29 | .345 | 29.0 | — | — |
| 24 | Strong, v7h CFG $w=4$ | 79.80 | .363 | 130 | .067 | **30.3** | — | — |
| 24 | **bucket:16 belief** | 173.39 | .425 | 142 | .028 | **18.2** | **60.41** | effective |
| 24 | bucket:32 belief | 60.16 | .307 | 86 | .097 | **31.3** (> strong) | 101.67 (worse than bare) | harmful |
| 24 | *after guidance*: bk16 $w$2 / composed | 60.41 / 48.70 | .308 / .323 | 102 / 95 | .087 / .083 | 34.7 / 29.6 | — | — |

**Figure A6.** Per-resolution normalised TV-vs-FD plot (`fig_tv_vs_fd_r.png`): $x = \mathrm{TV}_{\text{ref}}/\mathrm{TV}_{\text{strong}}$, $y = \mathrm{FD}_{\text{guided}}/\mathrm{FD}_{\text{bare}}$ for the ten bucket and snapshot references at 16 / 20 / 24 px. All same-caption beliefs with $x < 1$ give $y < 1$; all references with $x \ge 1$ give $y \approx 1$ or $y > 1$.

### A.9 16-colour quantisation at 20 / 24 / 32 px

Quantising every sample to 16 colours before FD (seed 0) helps the bare model at the $w = 4$ default substantially (21.98 → 12.64 / 45.92 → 42.65 / 79.80 → 64.38 / 96.85 → 81.41 at 16 / 20 / 24 / 32 px; q16 of the best-CFG samples at 16 px, $w = 1.5$: 12.24 → 11.55, a much smaller gain, because there is less colour bleeding left to remove) and does not help, or hurts, the guided rows (label reference 8.59 → 8.99 / 32.89 → 37.31 / 60.41 → 62.85 / 77.45 → 85.75; autoguidance 8.98 → 7.82 / 31.78 → 35.72 / 53.36 → 53.00 / 75.23 → 76.79; composed 7.67 → 8.34 / 29.66 → 34.09 / 48.70 → 51.37 / 69.27 → 74.57). At 16 px autoguidance and composed swap order after q16 (7.82 vs 8.34), and the composed margin over the $w = 4$ default shrinks from −35 / −39 / −28 % to −20 / −20 / −8 % at 20 / 24 / 32 px. Part of the gain is in the colour domain — bleeding that a 16-colour palette partly removes — and that part grows with resolution.

### A.10 Inception-v3 clean-FID / KID in full

On natural-image features after nearest-neighbour upsampling to 64 px, the composed reference is best or tied-best at every resolution and on both models (FID 7.97 vs autoguidance 7.84 at 16 px, within the 3-seed spread; 9.62 vs 9.85, 13.26 vs 13.83, 17.24 vs 18.56 at 20 / 24 / 32 px), and higher-bucket references are harmful here too (11.26 / 15.45 / 22.56 / 25.83 vs bare 9.57 / 11.82 / 16.13 / 19.77). The label-only reference is Inception-weak: it lowers FID at 16 and 20 px (9.57 → 8.61, 11.82 → 10.75) but is flat or slightly worse at 24 and 32 px (16.13 → 16.45, KID 2.40 → 3.19; 19.77 → 20.10, KID 2.34 → 3.31), and on the second model it lags autoguidance (9.77 vs 8.25). Rank agreement between the metric families is high (Spearman .98 / .86 / .96 / .68 / .83 at 12 / 16 / 20 / 24 / 32 px; n = 9 / 121 / 18 / 17 / 6), so this is a difference in sensitivity, not a contradiction. Inception FID / KID of best-CFG: @16 $w$1.5 9.31 / 1.88, $w$2 8.66 / .89, $w$1 10.65 / 3.01, $w$3 8.86 / .67, PAG-mid $w$2 8.92 / 1.31; @12 $w$2 6.45 / 1.00; @20 $w$2.5 10.66 / .67; @24 $w$2 15.34 / 2.64; @32 $w$2 19.37 / 2.77. Composed∘`bucketu` at 20 / 24 / 32 px: 9.84 / .33, 13.24 / 1.08, 17.61 / 1.86.

**Table A8.** Inception-v3 clean-FID / KID (×10⁻³), white composite, NEAREST ×4 to 64 px, same samples as Tables 1–3. Held-out real floor: FID 4.93 / 6.72 / 8.23 / 8.99 / 9.75 at 12 / 16 / 20 / 24 / 32 px. 16 px FID: seeds 0 / 1 / 2.

| $R$ | Bare CFG $w=4$ | Best CFG (DINOv2-optimal $w$) | Lower-bucket ref $w=2$ | Autoguidance 10 k | Composed | Reverse (higher bucket) |
|---|---|---|---|---|---|---|
| 16 (3 seeds, FID) | 9.57 / 9.61 / — | 9.31 ($w$1.5); 8.66 ($w$2) | bk12 8.61 / 8.63 / 8.64 | **7.84** / 7.97 / 7.71 | 7.97 / 7.86 / 7.95 | bk24 11.26, bk64 11.29 |
| 16 (KID) | 1.12 | 1.88 ($w$1.5); .89 ($w$2) | .78 | .55 | .63 | 2.71 |
| 20 (FID / KID) | 11.82 / 1.01 | 10.66 / .67 ($w$2.5) | bk16 10.75 / 1.03 | 9.85 / .58 | **9.62 / .26** | bk32 15.45 / 3.71 |
| 24 (FID / KID) | 16.13 / 2.40 | 15.34 / 2.64 ($w$2) | bk16 **16.45 / 3.19** (no gain) | 13.83 / 1.96 | **13.26 / 1.46** | bk32 22.56 / 7.55 |
| 32 (FID / KID) | 19.77 / 2.34 | 19.37 / 2.77 ($w$2) | bk24 **20.10 / 3.31** (no gain) | 18.56 / 2.91 | **17.24 / 1.81** | bk48 25.83 / 7.58 |
| v7s @16 (FID) | 10.33 | — | bk12 9.77 | **8.25** | 8.49 | bk20 12.86 |
| v7s @20 / @24 (FID) | 13.06 / 16.82 | — | 11.73 / 17.85 | 10.42 / 14.66 | **10.22 / 13.85** | — |

### A.11 Cost

**Table A9.** Wall-clock and peak memory for 1,000 samples @16 px (A100-80GB, exclusive; v7h 72.5 M; DDPM 100 steps; batch 500; includes model load).

| Configuration | NFE / step | s / 1,000 samples | Peak memory (MiB) |
|---|---|---|---|
| No guidance ($w=1$) | 1 | 42.5 | 4951 |
| CFG (measured at $w=4$; cost is independent of $w$) | 2 | 73.8 | 4951 |
| Label reference `bucket:12`, $w=2$ | 2 | 72.7 | 4951 |
| Autoguidance (snapshot 10 k), $w=1.5$ | 2 | 73.3 | 5507 |
| Composed, $w=1.5$ | 2 | 74.0 | 5507 |
| Composed + `cfg_text` 1.5 (alignment variant, Table A3) | 3 | 105.9 | 5507 |

The label reference costs what CFG costs (same weights, same memory, 1.5 % faster wall-clock); snapshot-based references hold one extra copy of the weights (+556 MiB) at the same wall-clock; the 3-NFE alignment variant is +43 %.

### A.12 Sampler robustness

The composed reference is not an artefact of the 100-step sampler. With 50 DDPM steps (16 px, seed 0) the ordering is unchanged and the margins widen: CFG $w = 4$ 20.81, best-CFG $w = 1.5$ 12.14 (CLIP 29.73), autoguidance 9.11, composed **6.63** (CLIP 29.51; an independent repeat of the same configuration gave 6.81), against 21.98 / 12.24 / 8.98 / 7.67 at 100 steps; the CFG optimum does not move with the step count while the composed margin over it grows from −37 % to −45 %. With 200 steps the composed reference gives 9.38, so fewer steps are slightly *better* for it (the 50-step result awaits a seed check). DDIM-50 is unusable for this model irrespective of guidance (bare 204.42), so DDIM rows are not reported.

### A.13 Falsified second components on top of the composed reference

Success criterion set in advance: FD < 7.2 at 16 px (seed 0). None of the components reaches it. The correction direction is a radial, low-frequency restoration of per-image contrast, so APG (removes the radial component), FDG with $w_{\text{low}} < w_{\text{high}}$ (down-weights low frequencies; mean term rises monotonically 2.5 → 3.5 → 4.8) and momentum all remove exactly the useful part. Interval scheduling shows the snapshot reference acts only in the low-noise half ($[0, .5]$ ≈ full schedule; $[.5, 1]$ ≈ single bucket:12 reference). Also falsified as components (single-reference autoguidance): channel-decoupled weights (RGB $w$ / alpha $w$): 2/1 → 11.16, 2/3 → 10.74, 3/2 → 32.36, 1.5/2.5 → 8.35, 2.5/1.5 → 19.17 — FD is set almost entirely by the RGB weight; degraded-input references: 2 × 2 box-blurred $x_t$ → 223.07 ($w = 2$) / 99.89 ($w = 1.5$), 1-px cyclic shift → 18.95.

**Table A10.** Second components on top of the composed reference @16 px, seed 0.

| Component | Configuration | FD | +q16 | mean / cov term |
|---|---|---|---|---|
| — (reference) | composed $w = 1.5$ | **7.67** | 8.34 | 2.5 / 5.2 |
| — | composed $w = 2$ | 10.83 | 10.14 | — |
| APG [cite: Sadat2024apg], RGB-only projection | composed $w = 1.5$, η = 0, β = −0.5 | 9.94 | 8.85 | — |
| APG RGB-only | composed $w = 2$ | 12.68 | 8.24 | — |
| APG full (4-channel projection) | composed $w = 2$ | 13.42 | 7.83 | — |
| APG projection only (β = 0) | composed $w = 2$ | 10.89 | 8.78 | — |
| APG momentum only (η = 1, β = −0.5) | composed $w = 2$ | 13.15 | 8.03 | — |
| APG RGB-only | composed $w = 3$ | 37.14 | 10.53 | — |
| APG RGB-only, single reference | bucket:12 $w = 2$ (no APG: 8.59) | 9.84 | 8.15 | — |
| Interval scheduling of snapshot ref | snapshot only $t/T \in [0, .5]$, $w = 1.5$ | 7.70 | — | 2.42 / 5.28 |
| Interval scheduling | snapshot only $[.5, 1]$, $w = 1.5$ | 8.99 | — | 3.54 / 5.45 |
| Interval scheduling | snapshot only $[.2, .8]$, $w = 1.5$ | 8.84 | — | 3.59 / 5.24 |
| Interval scheduling | snapshot only $[.2, .8]$, $w = 2$ | 9.04 | — | 2.70 / 6.34 |
| FDG [cite: Sabour2025], 1-level, $w_{\text{high}}$ / $w_{\text{low}}$ | 1.5 / 1.0 | 10.03 | — | 4.8 / 5.22 |
| FDG | 1.5 / 1.25 | 8.96 | — | 3.54 / 5.42 |
| FDG | 2 / 1.0 | 11.80 | — | 5.57 / 6.23 |
| FDG | 2 / 1.25 | 9.82 | — | 3.66 / 6.16 |
| FDG | 2.5 / 1.0 | 18.33 | — | 9.91 / 8.42 |

### A.14 Why higher-bucket references fail (moved from §5)

A higher bucket produces a belief with contrast at or above the strong prediction, so $e_{\text{strong}} - e_{\text{ref}}$ points *towards* lower contrast and extrapolating along it removes detail. The signature is over-guidance contraction: at 16 px the bucket:20/24/64 references raise precision to .935 and lower recall to .85–.88 (bare .907 / .902), and their samples are worse after 16-colour quantisation (q16 18.64 / 21.02 / 24.56 vs 12.64 bare) although raw FD is slightly better than CFG at the $w = 4$ default (17.14 / 17.27 / 19.34 vs 21.98) and clearly worse than best-CFG (12.24; all seed 0). At 12, 20, 24 and 32 px the higher bucket is no better than or worse than CFG at the default (seed-0 reverse rows 14.45; 46.62 and 59.13; 101.67; 113.93 against 12.91; 45.92; 79.80; 96.85), and worse than best-CFG at every resolution (9.71; 39.53; 67.34; 83.65); the second model repeats the pattern (bucket:20 25.31 raw vs 28.00 at $w = 4$ and 19.10 at best-CFG, 24.14 vs 13.60 after q16; precision .925 / recall .848). The effect is ordered by the label, which rules out the reading that any wrong label acts as an unconditional-like reference (as in ICG); mixing the wrong label with the unconditional branch (bucketmix, 9.42) sits between the label reference (8.59) and CFG (12.24 at its best weight, 21.98 at $w = 4$).

### A.15 An operational rule for reference selection (moved from Method)

Section 5 supports a selection rule that needs no training and no second network, only statistics of the reference sampled on its own (`--cfg 0`). Let TV be the mean adjacent $|\Delta\mathrm{RGB}|$ over opaque pixels of the pure-reference samples, compared with the strong model's samples at the same resolution.

1. **Lower TV than the strong model is necessary.** At 16 px (strong TV 32.0) the effective references have TV 21.6 (bucket:12) and 27.3 (snapshot); the ineffective ones 31.9 (bucket:24, guided FD 17.27) and 40.9 (bucket:64, 19.34). Likewise at 20 px (strong 31.0; bucket:16 21.8 effective; bucket:24 28.5 ineffective) and 24 px (strong 30.3; bucket:16 18.2 effective; bucket:32 31.3 harmful).
2. **Structure alignment is also necessary.** The reference must be a same-caption, same-input belief whose other statistics (colour count, opacity, spatial structure) match the strong prediction. Low TV from explicit degradation — a trained block-average branch (TV 14.2, harmful) or a trained contrast-shrunk branch (TV 15.9; 14.71 vs 9.43 for the free label on the same weights) — does not substitute, nor does a scalar contrast reduction of the strong prediction itself (`shrink:f`: 74.8 / 200.4 / 374.6 for $f = 0.85 / 0.7 / 0.5$ at $w = 2$; 444.4 for $f = 0.7$ at $w = 3$).
3. **Weight.** The label reference's $w$-curve is flat (worst / best ≈ 1.6× over $w \in [1.5, 3]$ at 16 px and ≤ 1.25× over $w \in [1.25, 3]$ at 20 / 24 px), the snapshot's steep (1.86× / 1.59× at 20 / 24 px; 8.98 → 30.11 at 16 px), and CFG needs per-resolution tuning (best weight 1.5 / 2.5 / 2 / 2 at 16 / 20 / 24 / 32 px; the recipe default $w = 4$ costs +72 % FD at 16 px but only +16–19 % at 20–32 px; on v7s the best weight is 2 / 2.5 / 2 at 16 / 20 / 24 px). A single fixed $w$ ($2$ for the label, $1.5$ with the snapshot) is practical for the guided rows.

The bucket label satisfies both conditions for free: it is a monotone contrast knob on the same network (pure-belief TV 21.6 → 31.9 → 40.9 for bucket 12 → 24 → 64 at 16 px) while colour count stays close to the strong model (78 vs 73) and the layout is that of the same denoising trajectory.

### A.16 A contaminated third model

v7_lowres, trained on data that included the evaluation sprites, shows the same relative gain for the label reference (CFG $w = 4$ 16.66 / 15.29 → `bucket:12` 7.40 / 7.08, seeds 0 / 1; −56 %) but is memorisation-contaminated and is reported here only.

### A.17 Broader impact

The method is a sampling-time guidance rule for an existing class of generative models; it adds no data, capacity or capability and costs the same as CFG (Table A9). Its intended application, small-sprite generation for games and interfaces, carries the usual concerns of generative art tools: displacement of pixel artists and reproduction of styles in the training corpus. Our training data are OGA-derived sprites [TODO cite OGA; TBD: state licence terms]; the evaluation sprites are excluded from training. Because the gain is mainly in colour and contrast statistics rather than semantics, and text adherence sits at the level of CFG at its own FD optimum rather than above it, the method is unlikely to raise the risk of targeted misuse relative to the underlying model. If it transfers to large bucketed models it would remove the need to store a second network for guidance, a modest efficiency benefit rather than a new capability.
