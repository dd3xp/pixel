# 4 Experiments

All numbers in this section are copied from `paper_outline.md` (tables (a)–(j′); the 20/24 px weight sweeps from the table under Table (a″), diag_wsweep20 / diag_gpu3b; the CFG weight curves, PAG, `shrink:f` and 50-step rows from Tables (j) and (j′)); entries the outline marks TBD are left as [TBD].

## 4.1 Setup

**Model and data.** Unless stated otherwise, every experiment uses one model, v7h: a 4-channel RGBA UNet (`UNet2DConditionModel`, 72.5 M parameters [TBD verify]) with a frozen CLIP text encoder and the target resolution supplied as a class embedding over the buckets $\{12,16,20,24,32,48,64\}$; $\epsilon$-prediction, 100-step DDPM, EMA weights. It is trained from random initialisation for 80 k steps on OGA-derived sprites with BLIP captions [TODO cite], each sprite also appearing as BOX-downsampled copies in the lower buckets; the training set is 16 px-dominated. EMA snapshots are saved every 5 k steps; the step-10 k snapshot serves as the autoguidance weak model. Architecture, data and recipe are not contributions and are identical across all rows.

**Held-out protocol.** A fixed set of 5,928 evaluation sprites is excluded from training (180,533 training rows remain). We generate one sample for each of 3,000 held-out captions (seed 0 unless stated) and compare against 3,000 real sprites passed through the same `to_tensor(R)` preprocessing as training (aspect-preserving, centred on a transparent canvas, hard alpha). The primary metric is *matched* Fréchet distance in DINOv2-small CLS feature space (FD-DINOv2; lower is better) at the native resolution $R$. The metric has a non-zero floor because the reference set is finite and disjoint from the held-out sprites: the held-out real sprites themselves score **3.37 / 3.45 / 12.14 / 12.96 / 13.77** at 12 / 16 / 20 / 24 / 32 px. Where indicated we also report FD after 16-colour quantisation of the samples (+q16), a secondary diagnostic that separates colour-domain from structural effects. Seed-to-seed sd of the protocol is 0.2–0.5 FD at 16 px; headline rows use three seeds (0, 1, 2), and differences of the order of the seed spread are not interpreted.

**Guidance rule.** With strong prediction $e_s$ and weak reference $e_w$, $e = e_w + w\,(e_s - e_w)$. Bare CFG uses the unconditional prediction. Its training-recipe default is $w=4$, but a sweep over $w \in \{1, 1.5, 2, 2.5, 3, 4, 7, 10\}$ (the full grid at 16 px; $\{1.5, 2, 2.5, 3, 4\}$ at the other resolutions, §4.3) shows that $w=4$ is over-guided at every resolution and on both models. The baseline throughout is therefore **best-CFG**: the CFG weight with the lowest FD at each resolution ($w = 1.5$ at 16 px, 2.5 at 20 px, 2 at 12 / 24 / 32 px for v7h), re-run with three seeds wherever it serves as a 3-seed baseline; the $w=4$ row is kept in every table and labelled "training default", and headline percentages are given against best-CFG first and against $w=4$ second in parentheses. The *label reference* uses the same weights, $x_t$ and caption but a lower bucket label (`bucket:12` at 16 px), $w=2$. Autoguidance [TODO cite Karras et al. 2024] uses the step-10 k snapshot under the correct label, $w=1.5$. The *composed* reference evaluates the snapshot under the lower label in a single weak forward, $w=1.5$. Unlike CFG, the guided rows use one fixed $w$ per reference type at every resolution and on both models (2 for the label, 1.5 whenever the snapshot is involved). All guided variants cost two network evaluations (NFE) per step, like CFG.

**Baselines and controls.** (i) Bare CFG with a weight sweep, CADS, interval-restricted CFG [TODO cite] and perturbed-attention guidance (PAG, [TODO cite Ahn et al. 2024]); (ii) autoguidance with a sweep over snapshot step and weight; (iii) *reverse* controls using a *higher* bucket label, and `bucketmix:12` (half lower bucket, half unconditional), which tests whether a wrong label merely acts as an unconditional-like reference [TODO cite ICG]; (iv) two *trained* degraded-view branches added to the same weights by fine-tuning with an extra label: `probe_cg`, targeting the 2×2 block-average of the real sprite, and `probe_cc`, targeting the sprite with RGB shrunk 0.6× towards its per-image mean, each evaluated against same-weights paired controls; (v) an *interventional* control without training, `shrink:f`, which uses the strong model's own $\hat{x}_0$ with contrast shrunk towards its per-image mean by a factor $f$ as the weak reference (1 NFE per step).

## 4.2 Main results at 16 px

Table 1 reports the 16 px results over three seeds. CFG at the training default $w=4$ is far from the floor (21.52 vs 3.45); re-tuning its weight to the FD optimum $w=1.5$ brings it to 12.49 ± 0.72 (seeds 12.24 / 13.32 / 11.91; CLIP 29.74 for seed 0), and no CFG weight goes below 12.2 (§4.3). Replacing the unconditional CFG reference with the model's own lower-bucket belief reduces FD to 8.52 ± 0.29 with no extra training or stored weights, 32 % below best-CFG (60 % below $w=4$), matching autoguidance with a stored snapshot (8.73 ± 0.26, −30 %; the 0.2 difference is within seed spread). Composing both references in one weak forward gives 7.53 ± 0.19, a 40 % reduction from best-CFG (65 % from $w=4$) and below every single-reference seed. Restricting the snapshot to the low-noise half ($t/T \le 0.5$) gives 7.63 ± 0.43, indistinguishable from the full composition.

**Table 1.** Matched FD-DINOv2 @16 px, v7h, 3,000 held-out captions, three seeds. Floor (held-out real sprites) = 3.45. +q16: FD after 16-colour quantisation (seeds 0 / 1 / 2). Best-CFG = the CFG weight with the lowest FD ($w=1.5$, from the sweep in Table 2); the $w=4$ row is the training-recipe default.

| Method | Weak reference | Extra training / storage | $w$ | seed 0 | seed 1 | seed 2 | FD mean ± sd | +q16 |
|---|---|---|---|---|---|---|---|---|
| Real held-out (floor) | — | — | — | 3.45 | — | — | **3.45** | — |
| CFG, training default | unconditional | none | 4 | 21.98 | 21.05 | 21.54 | 21.52 ± 0.47 | 12.64 / 12.68 / 12.55 |
| CFG, best weight (baseline) | unconditional | none | 1.5 | 12.24 | 13.32 | 11.91 | 12.49 ± 0.72 | 11.55 / — / — |
| Autoguidance | EMA snapshot 10 k | stores snapshot | 1.5 | 8.98 | 8.73 | 8.47 | 8.73 ± 0.26 | 7.82 / 7.98 / 7.82 |
| Label reference (ours) | `bucket:12`, same weights | **none** | 2 | 8.59 | 8.20 | 8.76 | 8.52 ± 0.29 | 8.99 / 8.78 / 8.57 |
| Composed (ours) | snapshot 10 k under `bucket:12` | stores snapshot | 1.5 | 7.67 | 7.32 | 7.61 | **7.53 ± 0.19** | 8.34 / 8.02 / 8.42 |
| Composed, snapshot only for $t/T\in[0,0.5]$ | as above; final weights elsewhere | stores snapshot | 1.5 | 7.70 | 7.16 | 8.02 | 7.63 ± 0.43 | [TBD] / 8.16 / 8.44 |

After 16-colour quantisation the ordering between autoguidance and the composed reference reverses (7.8 vs 8.3–8.4), so part of the composed gain lives in the colour domain; we return to this in §4.9 and in the Limitations. The FD decomposition for seed 0 (Appendix, Table (a′); CFG weight curve in Table 11) shows that the label reference has the lowest *mean* term (2.55 vs 12.13 for CFG at $w=4$, 5.87 at its best weight $w=1.5$, and 3.88 for autoguidance) while the snapshot reference has the best coverage (0.928 vs 0.892); the composition keeps the label's mean term (2.40) and most of the snapshot's covariance gain (cov 5.27, between 5.09 and 6.04; coverage 0.909).

## 4.3 Guidance-weight sweeps and other zero-training baselines

Table 2 collects the CFG weight curves at every resolution, the other zero-training CFG variants at 16 px (seed 0), and the sweeps of both single references at 16, 20 and 24 px (seed 0). (i) *CFG is over-guided at its default and cannot be re-tuned into the guided rows' range.* At 16 px the CFG curve is U-shaped: 16.39 / 12.24 / 12.98 / 16.67 / 21.98 / 42.95 / 62.12 for $w = 1 / 1.5 / 2 / 3 / 4 / 7 / 10$, with its minimum at $w=1.5$ (12.49 ± 0.72 over three seeds), so the recipe default $w=4$ is over-guided by 72 % and re-tuning alone removes 42 % of its FD; CADS and interval-restricted CFG are worse than plain CFG at the same weight. No CFG weight reaches below 12.2, whereas every reference-guided row in Table 1 is below 9. The optimum shifts with resolution — $w = 2.5$ at 20 px (39.57 ± 0.04 over three seeds), 2 at 24 px (67.77 ± 0.77), 2 at 32 px (83.65) and 2 at 12 px (9.71) — and the over-guidance of $w=4$ is much smaller there (re-tuning gains −16 / −14 / −14 % at 20 / 24 / 32 px vs −42 % at 16 px); on the second model v7s the best weights are 2 / 2.5 / 2 at 16 / 20 / 24 px (19.10 / 56.30 / 93.78; §4.5). CFG therefore needs per-resolution tuning, while the guided rows use one fixed $w$ per reference type throughout. *PAG* [TODO cite Ahn et al. 2024] (identity self-attention in the mid block, `pag:mid`) gives 12.57 / 12.62 / 12.90 at $w = 1.5 / 2 / 3$ (CLIP 29.45 / 29.41 / 29.45), 13.05 with mid + `down_blocks.1` at $w=2$, and 11.73 (CLIP 29.74) when a CFG term of 1.5 is added (3 NFE): PAG behaves as a gentler CFG — it tracks the CFG optimum and does not explode at $w=3$ — but does not remove the systematic error. (ii) The label reference has a flat weight curve (10.21 → 8.59 → 11.20 → 13.87 for $w = 1.5 \ldots 3$), whereas the snapshot reference explodes (8.98 → 30.11 over the same range). The same shape holds at 20 and 24 px: every weight in $w \in \{1.25, \ldots, 3\}$ for the label reference beats the $w=4$ default (45.92 at 20 px, seed 0; 79.80 at 24 px, seed 0) and every weight in $[1.5, 3]$ at 20 px and $[1.5, 2.5]$ at 24 px beats best-CFG (39.53 / 67.34, seed 0), with worst-over-sweep / best = 1.25× (20 px) and 1.23× (24 px), whereas the snapshot reference has a sharp optimum and explodes past it (1.86× at 20 px; at $w \ge 2.5$ it is worse than best-CFG at 20 px). At 24 px the snapshot optimum moves to $w=2$ (52.52 vs 53.36), so the composed row at $w=1.5$ is not tuned in the snapshot's favour. (iii) Degraded-*input* references fail outright (box-blurred $x_t$: 223.07; 1-px cyclic shift: 18.95), and channel-decoupled weights show that FD is set almost entirely by the RGB weight.

**Table 2.** Weight sweeps and zero-training baselines, v7h, seed 0 unless a ± is given (3 seeds). 16 px unless stated; best-CFG values in bold in the CFG rows. 20 / 24 px reference sweeps against best-CFG 39.53 / 67.34 (seed 0; 39.57 ± 0.04 / 67.77 ± 0.77 over three seeds) and the $w=4$ default 45.92 / 79.80 (seed 0; 47.04 ± 1.11 / 78.96 ± 0.74).

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
| CFG + CADS, 16 px | $w=4$ / $w=7$ / $w=7$, $s=.25$ | 44.58 / 71.17 / 88.61 |
| CFG $w=7$, interval $[0,.8]$, 16 px | — | 32.57 |
| PAG `pag:mid` [Ahn 2024], 16 px | 1.5 / 2 / 3 | 12.57 / 12.62 / 12.90 |
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
| **Label reference `bucket:16`, 20 px** | 1.25 / 1.5 / 2 / 2.5 / 3 | 40.96 / 37.48 / **32.89** / 35.15 / 38.53 |
| Autoguidance (snapshot 10 k), 20 px | 1.25 / 1.5 / 2 / 2.5 / 3 | 36.75 / **31.78** / 33.66 / 42.02 / 59.04 |
| **Label reference `bucket:16`, 24 px** | 1.25 / 1.5 / 2 / 2.5 / 3 | 71.12 / 63.09 / **60.41** / 65.77 / 74.61 |
| Autoguidance (snapshot 10 k), 24 px | 1.25 / 1.5 / 2 / 2.5 / 3 | 64.85 / 53.36 / **52.52** / 66.53 / 83.41 |

## 4.4 Generalisation across resolutions

Table 3 repeats the comparison at 12, 20, 24 and 32 px with the same weights and snapshot, against best-CFG re-tuned at each resolution ($w = 2.5$ at 20 px, 2 at 12 / 24 / 32 px; three seeds at 20 and 24 px) and against the $w=4$ default. The ordering bare > label reference ≈ autoguidance > composed holds at every resolution that has a lower bucket (at 32 px autoguidance, 75.23, edges the label reference, 77.45, seed 0), and the composed − autoguidance gap is 3.9–8.2 times the larger of the two seed sd (4.6 / 3.9 / 8.2 at 16 / 20 / 24 px). Relative to best-CFG the composed reference gains −40 / −27 / −29 / −17 % at 16 / 20 / 24 / 32 px (−65 / −39 / −39 / −28 % relative to $w=4$), and every gap is at least 7 times the larger of the two seed sd where three seeds exist (6.9 / 15.5 / 25 at 16 / 20 / 24 px). The *label-only* reference gains −32 % at 16 px but only −17 / −12 / −7 % at 20 / 24 / 32 px (autoguidance −30 / −20 / −20 / −10 %): the free variant matches autoguidance at 16 px and remains a component of the composed method, but its stand-alone advantage over a re-tuned CFG shrinks with resolution. At 12 px, which has no lower bucket, best-CFG is $w=2$ (9.71, seed 0) and autoguidance gives 8.54 ± 0.73 (−12 %; −34 % vs $w=4$). The bare model is much further from the floor at 20 / 24 / 32 px (best-CFG gaps 27.4 / 54.8 / 69.9 vs 9.0 at 16 px; at $w=4$, seed 0: 33.8 / 66.8 / 83.1 vs 18.5), reflecting the 16 px-dominated training data. The choice of lower bucket matters: at 24 px the nearest lower bucket (`bucket:20`, 66.20, seed 0) is the worst of the three, the reported `bucket:16` (59.32 ± 1.37) was chosen on FD, and the farther `bucket:12` is slightly better still (57.70 vs 60.41, seed 0); at 32 px the farther bucket is clearly worse (`bucket:16` 91.26 vs `bucket:24` 77.45, seed 0); §5 ties this to structural alignment. 12 px has no lower bucket, so only autoguidance applies.

**Table 3.** Matched FD-DINOv2 at native resolution, v7h. Mean ± sd over three seeds where available (3 s); otherwise seed 0. "Best CFG" = the CFG weight with the lowest FD at that resolution (weight in parentheses; sweep in Table 2); "CFG $w=4$" = training default. "Lower bucket used" = the nearest lower bucket except at 24 px, where bk16 is used and the nearest (bk20) is listed under "Other lower bucket", $w=2$. Δ columns are relative to best-CFG; the value in parentheses in the last column is relative to $w=4$.

| $R$ | Floor | CFG $w=4$ (default) | Best CFG ($w$) | Lower bucket used, $w=2$ | Other lower bucket, $w=2$ | Autoguidance 10 k, $w=1.5$ | Composed, $w=1.5$ | Δ vs best CFG: label / autog. / composed (composed vs $w=4$) |
|---|---|---|---|---|---|---|---|---|
| 12 | 3.37 | 13.02 ± 0.28 (3 s) | 9.71 (2) | — (none exists) | — | **8.54 ± 0.73** (3 s) | — | — / −12 % / — (autog. −34 %) |
| 16 | 3.45 | 21.52 ± 0.47 (3 s) | 12.49 ± 0.72 (1.5; 3 s) | bk12 8.52 ± 0.29 (3 s) | — | 8.73 ± 0.26 (3 s) | **7.53 ± 0.19** (3 s) | −32 / −30 / **−40 %** (−65 %) |
| 20 | 12.14 | 47.04 ± 1.11 (3 s) | 39.57 ± 0.04 (2.5; 3 s) | bk16 32.75 ± 0.60 (3 s) | bk12 35.85 | 31.57 ± 0.29 (3 s) | **28.90 ± 0.69** (3 s) | −17 / −20 / **−27 %** (−39 %) |
| 24 | 12.96 | 78.96 ± 0.74 (3 s) | 67.77 ± 0.77 (2; 3 s) | bk16 (not nearest) 59.32 ± 1.37 (3 s) | bk20 (nearest) 66.20; bk12 57.70 | 54.15 ± 0.72 (3 s) | **48.23 ± 0.43** (3 s; bk16 + 10 k) | −12 / −20 / **−29 %** (−39 %) |
| 32 | 13.77 | 96.85 | 83.65 (2) | bk24 77.45 | bk16 91.26 | 75.23 | **69.27** (bk24 + 10 k) | −7 / −10 / **−17 %** (−28 %) |

## 4.5 A second, independently trained model

To check that the effect is not specific to one training run, we train v7s with the same recipe, data and evaluation exclusions but a narrower width (41.3 M vs 72.5 M parameters), a different seed and 60 k steps; its own step-10 k EMA serves as snapshot. Its CFG curve has the same shape (16 px: 19.29 / 19.10 / 20.34 / 22.96 / 28.00 for $w = 1.5 / 2 / 2.5 / 3 / 4$; 20 px: 61.62 / 57.48 / 56.30 / 59.73 / 63.32; 24 px: 99.23 / 93.78 / 93.96 / [TBD] / 99.47; seed 0), with best-CFG at $w = 2 / 2.5 / 2$ (19.10 / 56.30 / 93.78) and a smaller over-guidance of the default than on v7h (re-tuning alone −32 / −11 / −6 %). Table 4 shows the same ordering at 16, 20 and 24 px: the composed reference is −45 / −35 / −36 % below best-CFG (−62 / −43 / −39 % below $w=4$; seed 0; 16 px seed 1: 10.28, −46 % vs the seed-0 best-CFG and −63 % vs its own $w=4$ row), the label reference −33 / −19 / −17 % and autoguidance −27 / −30 / −29 %, and the FD decomposition mirrors v7h: mean term 16.48 → 5.50 (label) / 7.04 (autoguidance) / 3.84 (composed), covariance term 11.51 → 7.31 / 6.91 / 6.70. A third model, v7_lowres, trained on data that included the evaluation sprites, shows the same relative gain (16.66 → 7.40 and 15.29 → 7.08, seeds 0 / 1) but is contaminated and appears only in the appendix.

**Table 4.** Second model v7s (clean), matched FD-DINOv2. 16 px: seed 0 / seed 1; 20 and 24 px: seed 0. Best CFG = lowest-FD CFG weight (in parentheses; seed 0). v7h row repeated from Tables 1–2 (seed 0) for reference. Δ = composed vs best CFG (vs the $w=4$ default in parentheses).

| Model | CFG $w=4$ (default) | Best CFG ($w$) | Label reference, $w=2$ | Autoguidance 10 k, $w=1.5$ | Composed, $w=1.5$ | Reverse `bucket:20`, $w=2$ | Δ composed vs best CFG (vs $w=4$) |
|---|---|---|---|---|---|---|---|
| v7h @16 (72.5 M) | 21.98 | 12.24 (1.5) | bk12 8.59 | 8.98 | **7.67** | 17.14 | −37 % (−65 %); 3-seed −40 % |
| v7s @16 (41.3 M) | 28.00 / 27.75 | 19.10 (2) | bk12 12.80 / 12.74 | 13.95 / 15.17 | **10.54 / 10.28** | 25.31 | −45 % / −46 % (−62 % / −63 %) |
| v7s @20 | 63.32 | 56.30 (2.5) | bk16 45.81 | 39.13 | **36.31** | — | −35 % (−43 %) |
| v7s @24 | 99.47 | 93.78 (2) | bk16 78.18 | 66.83 | **60.20** | — | −36 % (−39 %) |

## 4.6 Reverse controls: the reference must be a *lower* bucket

If a wrong label merely acted as a generic unconditional-like reference, a higher bucket would help as much as a lower one. It does not (Table 5, seed 0). At 12, 20 (bk32), 24 and 32 px the higher-bucket reference is *worse than CFG at the training default* (+1.5, +13.2, +21.9, +17.1 FD against the seed-0 $w=4$ rows; against the 3-seed means 13.02 ± 0.28 / 47.04 ± 1.11 / 78.96 ± 0.74 the reverse rows 14.45 / 59.13 / 101.67 remain above bare by many sd); at 20 px `bucket:24` is indistinguishable from the $w=4$ row (+0.7). Against best-CFG every reverse row is worse at every resolution, 16 px included: +4.7 (12 px), +4.9 / +5.0 / +7.1 (16 px), +7.1 / +19.6 (20 px), +34.3 (24 px), +30.3 (32 px). At 16 px higher buckets are slightly better than the $w=4$ default in raw FD (17.14–19.34 vs 21.98) but worse than best-CFG (12.24) and worse after quantisation (q16 18.64 / 21.02 / 24.56 vs 12.64), with precision rising to .935 and recall falling to .85–.88, the signature of over-guidance contraction; v7s shows the same (`bucket:20`: 25.31 vs best-CFG 19.10; q16 24.14, precision .925, recall .848). `bucketmix:12` (9.42) lies between the label reference (8.59) and CFG (12.24 at its best weight, 21.98 at $w=4$; all seed 0), not at CFG.

**Table 5.** Reverse controls (higher bucket as weak reference, $w=2$), v7h, seed 0. Effects are given against the $w=4$ default and, after the slash, against best-CFG.

| $R$ | CFG $w=4$ (default) | Best CFG | Best lower / composed | Higher-bucket reference | Effect vs $w=4$ / vs best CFG |
|---|---|---|---|---|---|
| 12 | 12.91 | 9.71 | — / autoguidance 9.18 | bk16 **14.45** | worse (+1.5) / worse (+4.7) |
| 16 | 21.98 | 12.24 | bk12 8.59 / 7.67 | bk20 17.14; bk24 17.27; bk64 19.34 | slightly better raw, worse after q16 / worse (+4.9 … +7.1) |
| 20 | 45.92 | 39.53 | bk16 32.89 / 29.66 | bk24 **46.62**; bk32 **59.13** | ≈ bare (+0.7) / worse (+13.2) ; worse (+7.1) / worse (+19.6) |
| 24 | 79.80 | 67.34 | bk16 60.41 / 48.70 | bk32 **101.67** | worse (+21.9) / worse (+34.3) |
| 32 | 96.85 | 83.65 | bk24 77.45 / 69.27 | bk48 **113.93** | worse (+17.1) / worse (+30.3) |

## 4.7 Trained degraded-view branches

Table 6 compares the free label reference with two trained degraded-view branches on the same fine-tuned weights. The block-average branch (`probe_cg`) is *harmful* (21.53 vs bare 19.17 at $w=1.5$; 35.22 and 59.38 at $w=2, 3$); the contrast-shrunk branch (`probe_cc`) helps (14.71 vs 20.63) but stays far behind the label reference on the same weights (9.43) and autoguidance (8.77). At 12 px, where no lower bucket exists, `probe_cc` improves on bare (9.66 vs 11.78) but not on autoguidance (8.16). The bare column in Table 6 is CFG at $w=4$; no CFG weight sweep was run on the fine-tuned weights, so these rows are compared within the table only.

*Interventional control without training (`shrink:f`).* The trained contrast-shrunk branch changes the weights; the cleanest test of the reading "the lower-bucket belief is just a lower-contrast copy of the strong prediction" needs neither training nor a second forward. We take the strong model's own $\hat{x}_0$ at each step, shrink its RGB contrast towards the per-image mean by a factor $f$, and use it as the weak reference, so that the guidance update is a uniform contrast amplification of the strong prediction (1 NFE per step). It is monotonically harmful (v7h, 16 px, seed 0): $f = 0.5 / 0.7 / 0.85$ at $w=2$ give FD 374.6 / 200.4 / 74.8, and $f = 0.7$ at $w=3$ gives 444.4 (CLIP 28.5–29.5, R@1 5–12 %), against 12.24 for best-CFG and 8.59 for the `bucket:12` reference. The lower-bucket reference is therefore not a low-contrast *scalar*; its error direction is spatially structured (§5.5).

**Table 6.** Same-weights paired controls inside the trained probes, 16 px, seed 0 (12 px row: no lower bucket); last block: the untrained `shrink:f` control on v7h.

| Fine-tuned weights | Bare CFG $w=4$ | Trained branch, $w=1.5$ | `bucket:12`, $w=2$ | Autoguidance 10 k, $w=1.5$ |
|---|---|---|---|---|
| `probe_cg` (2×2 block average) | 19.17 (q16 12.37) | 21.53 (q16 16.05) | 10.12 (q16 9.51) | 9.10 (q16 8.45) |
| `probe_cc` (contrast shrink 0.6×) | 20.63 (q16 12.66) | **14.71** (q16 9.17) | 9.43 (q16 8.97) | 8.77 (q16 7.78) |
| `probe_cc` @12 px | 11.78 | 9.66 ($w=1.25$) / 10.07 ($w=1.5$) | — | 8.16 |
| v7h, `shrink:f` (untrained, 1 NFE): $f = 0.85 / 0.7 / 0.5$, $w=2$; $f=0.7$, $w=3$ | best CFG 12.24 ($w=4$: 21.98) | **74.8 / 200.4 / 374.6; 444.4** | 8.59 | 8.98 |

## 4.8 Sampler robustness

The composed reference is not an artefact of the 100-step sampler. With 50 DDPM steps (16 px, seed 0) the ordering is unchanged and the margins widen: CFG $w=4$ 20.81, autoguidance 9.11, composed **6.63** (an independent repeat of the same configuration, run on a different GPU, gave 6.81; mean 6.72), against 21.98 / 8.98 / 7.67 at 100 steps ; best-CFG $w=1.5$ at 50 steps gives 12.14 (100 steps 12.24), so the CFG optimum does not move with the step count while the composed margin over it grows from −37 % to −45 %. With 200 steps the composed reference gives 9.38, so fewer steps are slightly *better* for it, consistent with extrapolation error accumulating over steps (the 50-step result awaits a seed check). DDIM-50 is unusable for this model irrespective of guidance (bare 204.42), so DDIM rows appear only in the appendix with that caveat.

## 4.9 Alignment–fidelity frontier

Guidance against a same-caption reference removes the unconditional CFG term, and the guided rows score lower CLIP similarity than CFG at the training default. Table 7 quantifies this with CLIP ViT-B/32 similarity (100·cos) and retrieval R@1 among 1 + 99 captions (chance 1 %) on the *same* saved samples: at 16 px every reference-guided row, autoguidance included, sits at 29.37–29.58 against 30.06 for CFG $w=4$ (R@1 18 → 13–15 %), and the gap grows to −0.9 at 32 px; it is method-independent (snapshot and label references within 0.2), grows with $w$ (`bucket:12` $w=3$: 29.25), and is absent for reverse references. CFG at $w=4$ scores *above* the real sprites (30.06 vs 29.80), i.e. it over-aligns, while the guided rows sit ≈ 0.3 below real.

This is not, however, a cost specific to reference guidance; it is the price of any guidance weight that reaches the FD optimum. Along the CFG weight curve (Table 11, seed 0) CLIP rises monotonically with $w$ — 29.43 / 29.74 / 29.87 / 30.01 / 30.06 / 30.04 / 30.05 for $w = 1 / 1.5 / 2 / 3 / 4 / 7 / 10$ (R@1 13.8 → 18.3 %) — while FD traces a U (16.39 / 12.24 / 12.98 / 16.67 / 21.98 / 42.95 / 62.12). Lowering CFG to its own FD optimum $w=1.5$ costs exactly the CLIP the guided rows pay (29.74, R@1 15.9 %), and $w=4$ buys its extra 0.32 CLIP with +9.7 FD. The right comparison is therefore the FD–CLIP plane (`fig_pareto_fd_clip.png`, Fig. [TBD]). There, **every reference-guided row lies 3–5 FD below the best CFG point and further below the CFG curve at its own CLIP**: `bucket:12` $w=2$ at (8.59, 29.37) against CFG $w=1$ at (16.39, 29.43); autoguidance at (8.98, 29.58) and composed at (7.67, 29.51) against CFG $w=1.5$ at (12.24, 29.74) and $w=1$ at (16.39, 29.43), between which the CFG curve passes at FD ≈ 14–15 for CLIP 29.5–29.6. CFG cannot reach FD < 12.2 at any weight, and PAG tracks the CFG curve (12.57–12.90 at CLIP 29.41–29.45; 11.73 at 29.74 with an added CFG term). The "alignment cost" of the guided rows relative to $w=4$ is thus exactly the cost CFG itself pays when its weight is lowered to the FD optimum, and the guided rows deliver a lower FD at every CLIP level CFG can reach in the 29.4–29.7 range.

Two zero-training variants move along the frontier towards higher CLIP (Table 8): (a) `bucketu:12`, a reference under the lower bucket *and* the empty caption, so the guidance direction contains the CFG text direction at the same 2 NFE; (b) an additive plain-CFG term (`cfg_text`) on top of the reference term (3 NFE). Both the composed reference under `bucketu:12` at $w=1.5$ — FD 8.60, CLIP 29.77, the real-data level (29.80; R@1 16.1 % vs 16.4 %), 2 NFE — and composed + `cfg_text` 1.5 (9.19, 29.72; 3 NFE) *dominate* the best CFG point (12.24, 29.74): lower FD at equal or higher CLIP. None of the seven variants reaches the $w=4$ CLIP of 30.06; along the guided frontier (29.37 @ 8.59 → 29.77 @ 8.60 → 29.89 @ 11.10) every +0.1 CLIP costs roughly +0.5–1 FD, mostly in the mean term, and the same exchange is steeper for CFG (+3 FD per 0.1 CLIP between $w=1.5$ and $w=4$). The main tables keep the un-fixed composed row (7.67); the `bucketu` composed row is the operating point for real-level alignment at +0.9 FD (about 3 seed sd).

**Table 7.** CLIP ViT-B/32 100·cos (R@1 in parentheses) of the same samples as Tables 1 and 3; seed 0 / seed 1 where two values are given. "Best CFG" = the lowest-FD CFG weight of Table 3.

| $R$ | Real | CFG $w=4$ (default) | Best CFG | Lower-bucket ref, $w=2$ | Autoguidance, $w=1.5$ | Composed, $w=1.5$ | Higher-bucket ref |
|---|---|---|---|---|---|---|---|
| 12 | 29.55 (13.9 %) | 29.76 / 29.74 (15 %) | 29.65 (14.2 %) ($w=2$) | — | 29.41 / 29.39 (13 %) | — | bk16 29.32 (11.8 %) |
| 16 | 29.80 (16.4 %) | 30.06 / 30.04 / 30.03 (18 %) | 29.74 (15.9 %) ($w=1.5$, seed 0) | bk12 29.37 / 29.38 / 29.37 (13 %) | 29.58 / 29.53 / 29.57 (15 %) | 29.51 / 29.47 / 29.54 (14 %) | bk24 29.58; bk64 29.60 (15 %) |
| 20 | 29.86 (19.3 %) | 30.16 / 30.09 (20 %) | 30.02 (18.3 %) ($w=2.5$) | bk16 29.34 / 29.35 (14 %) | 29.45 / 29.52 (15–16 %) | 29.48 / 29.49 (15 %) | bk32 29.62 (16 %) |
| 24 | 29.81 (20.2 %) | 30.11 / 30.12 (20 %) | 29.89 (19.3 %) ($w=2$) | bk16 29.09 / 29.11 (13 %) | 29.35 / 29.38 (15 %) | 29.33 / 29.34 (15 %) | bk32 29.48 (16 %) |
| 32 | 29.65 (20.9 %) | 29.72 (21.5 %) | 29.43 (20.6 %) ($w=2$) | bk24 28.49 (14 %) | 28.69 (15 %) | 28.78 (15 %) | bk48 28.85 (17.5 %) |

At 20 / 24 / 32 px the best-CFG row keeps a 0.5–0.65 CLIP advantage over the composed row (30.02 vs 29.48, 29.89 vs 29.33, 29.43 vs 28.78), so the both-coordinate dominance shown at 16 px by composed$\circ$`bucketu:12` is, so far, a 16 px result; the corresponding `bucketu` variants at 20 / 24 / 32 px are [TBD: diag_review6]. What holds at every resolution is the frontier statement itself: the composed row's FD is 27–40 % below the best CFG point while its CLIP is 0.3–0.9 below the real-data value, a gap of the same size CFG pays between its own $w=4$ and $w=1.5$ operating points.

**Table 11.** The CFG weight curve and the alignment–fidelity frontier, 16 px, v7h, seed 0 (source: outline Table (j); plotted in `fig_pareto_fd_clip.png`). FD with mean / covariance terms, CLIP 100·cos and R@1 on the same saved samples.

| Row | NFE | FD | mean / cov | CLIP | R@1 |
|---|---|---|---|---|---|
| CFG $w=1$ | 2 | 16.39 | 9.47 / 6.92 | 29.43 | 13.8 % |
| CFG $w=1.5$ (**best CFG**) | 2 | **12.24** | 5.87 / 6.37 | 29.74 | 15.9 % |
| CFG $w=2$ | 2 | 12.98 | 6.32 / 6.66 | 29.87 | 16.8 % |
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

**Table 8.** Alignment variants along the frontier, 16 px, seed 0.

| Row | NFE | FD | +q16 | CLIP | R@1 |
|---|---|---|---|---|---|
| CFG $w=4$ (training default) | 2 | 21.98 | 12.64 | 30.06 | 18.3 % |
| CFG $w=1.5$ (best CFG) | 2 | 12.24 | 11.55 | 29.74 | 15.9 % |
| `bucket:12` $w=2$ | 2 | 8.59 | 8.99 | 29.37 | 13.1 % |
| Composed $w=1.5$ | 2 | **7.67** | 8.34 | 29.51 | 14.4 % |
| (a) `bucketu:12` $w=2$ / $w=1.5$ | 2 | 12.10 / 10.45 | 9.24 / 8.60 | 29.84 / 29.77 | 16.1 / 16.4 % |
| (a) composed under `bucketu:12`, $w=1.5$ / $w=1.25$ | 2 | 8.60 / 9.46 | 9.32 / 9.60 | 29.77 / 29.65 | 16.1 / 15.7 % |
| (b) `bucket:12` $w=2$ + `cfg_text` 1.5 | 3 | 9.50 | 8.71 | 29.71 | 16.2 % |
| (b) composed $w=1.5$ + `cfg_text` 1.5 / 2 | 3 | 9.19 / 11.10 | 8.57 / 9.48 | 29.72 / 29.89 | 15.4 / 15.9 % |

## 4.10 A second metric family

Because DINOv2 sees one patch per pixel at these resolutions, we also compute Inception-v3 clean-FID and KID [TODO cite clean-fid] on the same saved samples (white composite, NEAREST ×4 to 64 px). Inception has far less headroom here (floors 4.93 / 6.72 / 8.23 / 8.99 / 9.75 at 12–32 px; bare 9.57 vs floor 6.72 at 16 px). The bare rows of Table 9 are CFG at the $w=4$ default; the best-CFG column re-scores the FD-DINOv2-optimal CFG weight of Table 3 under Inception. The margin of the composed reference over best CFG is smaller under Inception than under DINOv2 — FID −8 / −10 / −14 / −11 % at 16 / 20 / 24 / 32 px (8.66 → 7.97, 10.66 → 9.62, 15.34 → 13.26, 19.37 → 17.24) against −40 / −27 / −29 / −17 % — but KID roughly halves at every resolution (0.89 → 0.63, 0.67 → 0.26, 2.64 → 1.46, 2.77 → 1.81), and the Inception-optimal CFG weight at 16 px is $w = 2$ (8.66; $w = 1 / 1.5 / 2.5 / 3$: 10.65 / 9.31 / — / 8.86), not the DINOv2-optimal 1.5, so the Inception column compares against CFG's own best weight under that metric. PAG (mid, $w=2$) scores 8.92 / 1.31 under Inception, again indistinguishable from same-weight CFG. Table 9 shows that the two central claims survive the change of feature space: the composed reference is best or tied-best at every resolution and on both models, and higher-bucket references are harmful. Under Inception, autoguidance and composed are within seed spread at 16 px; the composed advantage is clear at 20 / 24 / 32 px. The *label-only* reference, however, is DINOv2-visible but Inception-weak: it lowers FID at 16 / 20 px but is flat or slightly worse at 24 / 32 px (16.13 → 16.45, KID 2.40 → 3.19) and lags autoguidance on v7s (9.77 vs 8.25), in line with the q16 result (Appendix, Table (g′)): the label reference mainly corrects per-pixel colour and contrast statistics, which DINOv2 weights heavily and 64 px Inception barely sees. Rank agreement between the metric families over all saved rows is Spearman .98 / .86 / .96 / .68 / .83 at 12 / 16 / 20 / 24 / 32 px (Pearson .91–.99).

**Table 9.** Inception-v3 clean-FID (KID ×10⁻³ where given), same samples as Tables 1, 3, 4. 16 px FID: seeds 0 / 1 / 2.

| $R$ | Bare CFG ($w=4$ default) | Best CFG (DINOv2-optimal $w$) | Lower-bucket ref, $w=2$ | Autoguidance 10 k | Composed | Reverse (higher bucket) |
|---|---|---|---|---|---|---|
| 16, FID (3 seeds) | 9.57 / 9.61 / — | 9.31 ($w=1.5$); 8.66 ($w=2$) | bk12 8.61 / 8.63 / 8.64 | **7.84** / 7.97 / 7.71 | 7.97 / 7.86 / 7.95 | bk24 11.26; bk64 11.29 |
| 16, KID | 1.12 | 1.88 ($w=1.5$); .89 ($w=2$) | .78 | .55 | .63 | 2.71 |
| 20, FID / KID | 11.82 / 1.01 | 10.66 / .67 ($w=2.5$) | bk16 10.75 / 1.03 | 9.85 / .58 | **9.62 / .26** | bk32 15.45 / 3.71 |
| 24, FID / KID | 16.13 / 2.40 | 15.34 / 2.64 ($w=2$) | bk16 16.45 / 3.19 (no gain) | 13.83 / 1.96 | **13.26 / 1.46** | bk32 22.56 / 7.55 |
| 32, FID / KID | 19.77 / 2.34 | 19.37 / 2.77 ($w=2$) | bk24 20.10 / 3.31 (no gain) | 18.56 / 2.91 | **17.24 / 1.81** | bk48 25.83 / 7.58 |
| v7s @16, FID | 10.33 | — | bk12 9.77 | **8.25** | 8.49 | bk20 12.86 |
| v7s @20 / @24, FID | 13.06 / 16.82 | — | 11.73 / 17.85 | 10.42 / 14.66 | **10.22 / 13.85** | — |

## 4.11 Cost

Table 10 gives wall-clock and peak memory for 1,000 samples at 16 px on one A100-80GB (batch 500, 100 DDPM steps, model load included). The label reference costs exactly what CFG costs; snapshot-based references hold one extra copy of the weights (+556 MiB) at the same wall-clock; the 3-NFE alignment variant is +43 %.

**Table 10.** Cost per 1,000 samples @16 px, v7h (72.5 M).

| Configuration | NFE / step | s / 1,000 samples | Peak memory (MiB) |
|---|---|---|---|
| No guidance ($w=1$) | 1 | 42.5 | 4951 |
| CFG (measured at $w=4$; cost is independent of $w$) | 2 | 73.8 | 4951 |
| Label reference `bucket:12`, $w=2$ | 2 | 72.7 | 4951 |
| Autoguidance (snapshot 10 k), $w=1.5$ | 2 | 73.3 | 5507 |
| Composed, $w=1.5$ | 2 | 74.0 | 5507 |
| Composed + `cfg_text` 1.5 | 3 | 105.9 | 5507 |

## 4.12 Qualitative results

Figures `fig_qual_12px.png`, `fig_qual_16px.png`, `fig_qual_20px.png`, `fig_qual_24px.png` and `fig_qual_32px.png` show prompt-aligned samples at each resolution (16 px: 6 rows × 24 sprites, same seed and prompt per column; rows include bare CFG at $w=4$, autoguidance, `bucket:12` and composed [TBD: add a best-CFG $w=1.5$ row to the 16 px figure]). The bare model's characteristic failures are bleeding colours and insufficient local contrast, which the guided rows remove [TBD: confirm wording against the final figure]. Guided samples still have roughly 1.8× the colour count and one quarter of the flat-region fraction of real sprites, which accounts for the remaining gap to the floor (7.5 vs 3.45) analysed in §5.
