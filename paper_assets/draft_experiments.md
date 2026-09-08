# 4 Experiments

All numbers in this section are copied from `paper_outline.md` (tables (a)–(i); the 20/24 px weight sweeps from the table under Table (a″), diag_wsweep20 / diag_gpu3b); entries the outline marks TBD are left as [TBD].

## 4.1 Setup

**Model and data.** Unless stated otherwise, every experiment uses one model, v7h: a 4-channel RGBA UNet (`UNet2DConditionModel`, 72.5 M parameters [TBD verify]) with a frozen CLIP text encoder and the target resolution supplied as a class embedding over the buckets $\{12,16,20,24,32,48,64\}$; $\epsilon$-prediction, 100-step DDPM, EMA weights. It is trained from random initialisation for 80 k steps on OGA-derived sprites with BLIP captions [TODO cite], each sprite also appearing as BOX-downsampled copies in the lower buckets; the training set is 16 px-dominated. EMA snapshots are saved every 5 k steps; the step-10 k snapshot serves as the autoguidance weak model. Architecture, data and recipe are not contributions and are identical across all rows.

**Held-out protocol.** A fixed set of 5,928 evaluation sprites is excluded from training (180,533 training rows remain). We generate one sample for each of 3,000 held-out captions (seed 0 unless stated) and compare against 3,000 real sprites passed through the same `to_tensor(R)` preprocessing as training (aspect-preserving, centred on a transparent canvas, hard alpha). The primary metric is *matched* Fréchet distance in DINOv2-small CLS feature space (FD-DINOv2; lower is better) at the native resolution $R$. The metric has a non-zero floor because the reference set is finite and disjoint from the held-out sprites: the held-out real sprites themselves score **3.37 / 3.45 / 12.14 / 12.96 / 13.77** at 12 / 16 / 20 / 24 / 32 px. Where indicated we also report FD after 16-colour quantisation of the samples (+q16), a secondary diagnostic that separates colour-domain from structural effects. Seed-to-seed sd of the protocol is 0.2–0.5 FD at 16 px; headline rows use three seeds (0, 1, 2), and differences of the order of the seed spread are not interpreted.

**Guidance rule.** With strong prediction $e_s$ and weak reference $e_w$, $e = e_w + w\,(e_s - e_w)$. Bare CFG uses the unconditional prediction, $w=4$ (the training-recipe default; no stronger CFG setting we tested improves on it, §4.3, and a downward sweep over $w<4$ is in progress). The *label reference* uses the same weights, $x_t$ and caption but a lower bucket label (`bucket:12` at 16 px), $w=2$. Autoguidance [TODO cite Karras et al. 2024] uses the step-10 k snapshot under the correct label, $w=1.5$. The *composed* reference evaluates the snapshot under the lower label in a single weak forward, $w=1.5$. All guided variants cost two network evaluations (NFE) per step, like CFG.

**Baselines and controls.** (i) Bare CFG with a weight sweep, CADS and interval-restricted CFG [TODO cite]; (ii) autoguidance with a sweep over snapshot step and weight; (iii) *reverse* controls using a *higher* bucket label, and `bucketmix:12` (half lower bucket, half unconditional), which tests whether a wrong label merely acts as an unconditional-like reference [TODO cite ICG]; (iv) two *trained* degraded-view branches added to the same weights by fine-tuning with an extra label: `probe_cg`, targeting the 2×2 block-average of the real sprite, and `probe_cc`, targeting the sprite with RGB shrunk 0.6× towards its per-image mean, each evaluated against same-weights paired controls.

## 4.2 Main results at 16 px

Table 1 reports the 16 px results over three seeds. The clean baseline is far from the floor (21.52 vs 3.45). Replacing the unconditional CFG reference with the model's own lower-bucket belief reduces FD to 8.52 ± 0.29 with no extra training or stored weights, matching autoguidance with a stored snapshot (8.73 ± 0.26; the 0.2 difference is within seed spread). Composing both references in one weak forward gives 7.53 ± 0.19, a 65 % reduction and below every single-reference seed. Restricting the snapshot to the low-noise half ($t/T \le 0.5$) gives 7.63 ± 0.43, indistinguishable from the full composition.

**Table 1.** Matched FD-DINOv2 @16 px, v7h, 3,000 held-out captions, three seeds. Floor (held-out real sprites) = 3.45. +q16: FD after 16-colour quantisation (seeds 0 / 1 / 2).

| Method | Weak reference | Extra training / storage | $w$ | seed 0 | seed 1 | seed 2 | FD mean ± sd | +q16 |
|---|---|---|---|---|---|---|---|---|
| Real held-out (floor) | — | — | — | 3.45 | — | — | **3.45** | — |
| CFG (baseline) | unconditional | none | 4 | 21.98 | 21.05 | 21.54 | 21.52 ± 0.47 | 12.64 / 12.68 / 12.55 |
| Autoguidance | EMA snapshot 10 k | stores snapshot | 1.5 | 8.98 | 8.73 | 8.47 | 8.73 ± 0.26 | 7.82 / 7.98 / 7.82 |
| Label reference (ours) | `bucket:12`, same weights | **none** | 2 | 8.59 | 8.20 | 8.76 | 8.52 ± 0.29 | 8.99 / 8.78 / 8.57 |
| Composed (ours) | snapshot 10 k under `bucket:12` | stores snapshot | 1.5 | 7.67 | 7.32 | 7.61 | **7.53 ± 0.19** | 8.34 / 8.02 / 8.42 |
| Composed, snapshot only for $t/T\in[0,0.5]$ | as above; final weights elsewhere | stores snapshot | 1.5 | 7.70 | 7.16 | 8.02 | 7.63 ± 0.43 | [TBD] / 8.16 / 8.44 |

After 16-colour quantisation the ordering between autoguidance and the composed reference reverses (7.8 vs 8.3–8.4), so part of the composed gain lives in the colour domain; we return to this in §4.9 and in the Limitations. The FD decomposition for seed 0 (Appendix, Table (a′)) shows that the label reference has the lowest *mean* term (2.55 vs 12.13 for CFG and 3.88 for autoguidance) while the snapshot reference has the best coverage (0.928 vs 0.892); the composition keeps the label's mean term (2.40) and most of the snapshot's covariance gain (cov 5.27, between 5.09 and 6.04; coverage 0.909).

## 4.3 Guidance-weight sweeps and other zero-training baselines

Table 2 collects the weight sweeps at 16 px (seed 0) and the sweeps of both single references at 20 and 24 px (seed 0). (i) No stronger CFG setting improves on $w=4$: CFG at $w = 7, 10$, CADS and interval-restricted CFG are all worse than the baseline, so the residual error is not under-guidance (a sweep below $w=4$ is in progress). (ii) The label reference has a flat weight curve (10.21 → 8.59 → 11.20 → 13.87 for $w = 1.5 \ldots 3$), whereas the snapshot reference explodes (8.98 → 30.11 over the same range). The same shape holds at 20 and 24 px: every weight in $w \in \{1.25, \ldots, 3\}$ for the label reference beats bare CFG (45.92 at 20 px, seed 0; 78.96 ± 0.74 at 24 px), with worst-over-sweep / best = 1.25× (20 px) and 1.23× (24 px), whereas the snapshot reference has a sharp optimum and explodes past it (1.86× at 20 px). At 24 px the snapshot optimum moves to $w=2$ (52.52 vs 53.36), so the composed row at $w=1.5$ is not tuned in the snapshot's favour. (iii) Degraded-*input* references fail outright (box-blurred $x_t$: 223.07; 1-px cyclic shift: 18.95), and channel-decoupled weights show that FD is set almost entirely by the RGB weight.

**Table 2.** Weight sweeps and zero-training baselines, v7h, seed 0. 16 px unless stated; 20 / 24 px sweeps against bare CFG 45.92 / 78.96 ± 0.74 (seed 0 / 3 seeds).

| Reference | $w$ sweep | FD |
|---|---|---|
| CFG (unconditional), 16 px | 4 / 7 / 10 | 21.98 / 42.95 / 62.12 |
| CFG + CADS, 16 px | $w=4$ / $w=7$ / $w=7$, $s=.25$ | 44.58 / 71.17 / 88.61 |
| CFG $w=7$, interval $[0,.8]$, 16 px | — | 32.57 |
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
| Autoguidance (snapshot 10 k), 24 px | 1.25 / 1.5 / 2 / 2.5 / 3 | 64.85 / 53.36 / **52.52** / [TBD] / [TBD] |

## 4.4 Generalisation across resolutions

Table 3 repeats the comparison at 12, 20, 24 and 32 px with the same weights and snapshot. The ordering bare > label reference ≈ autoguidance > composed holds at every resolution that has a lower bucket (at 32 px autoguidance, 75.23, edges the label reference, 77.45, seed 0), and the composed − autoguidance gap is 3.9–8.2 times the larger of the two seed sd (4.6 / 3.9 / 8.2 at 16 / 20 / 24 px). Relative gains of the composed reference are −65 / −35 / −39 / −28 % at 16 / 20 / 24 / 32 px (seed 0). The bare model is much further from the floor at 20 / 24 / 32 px (gaps 33.8 / 66.8 / 83.1 vs 18.5 at 16 px), reflecting the 16 px-dominated training data. The choice of lower bucket matters: at 24 px the nearest lower bucket (`bucket:20`, 66.20, seed 0) is the worst of the three, the reported `bucket:16` (59.32 ± 1.37) was chosen on FD, and the farther `bucket:12` is slightly better still (57.70 vs 60.41, seed 0); at 32 px the farther bucket is clearly worse (`bucket:16` 91.26 vs `bucket:24` 77.45, seed 0); §5 ties this to structural alignment. 12 px has no lower bucket, so only autoguidance applies.

**Table 3.** Matched FD-DINOv2 at native resolution, v7h. Mean ± sd over three seeds where available (3 s); otherwise seed 0. "Lower bucket used" = the nearest lower bucket except at 24 px, where bk16 is used and the nearest (bk20) is listed under "Other lower bucket", $w=2$.

| $R$ | Floor | CFG $w=4$ | Lower bucket used, $w=2$ | Other lower bucket, $w=2$ | Autoguidance 10 k, $w=1.5$ | Composed, $w=1.5$ |
|---|---|---|---|---|---|---|
| 12 | 3.37 | 13.02 ± 0.28 (3 s) | — (none exists) | — | **8.54 ± 0.73** (3 s) | — |
| 16 | 3.45 | 21.52 ± 0.47 (3 s) | bk12 8.52 ± 0.29 (3 s) | — | 8.73 ± 0.26 (3 s) | **7.53 ± 0.19** (3 s) |
| 20 | 12.14 | 47.04 ± 1.11 (3 s) | bk16 32.75 ± 0.60 (3 s) | bk12 35.85 | 31.57 ± 0.29 (3 s) | **28.90 ± 0.69** (3 s) |
| 24 | 12.96 | 78.96 ± 0.74 (3 s) | bk16 (not nearest) 59.32 ± 1.37 (3 s) | bk20 (nearest) 66.20; bk12 57.70 | 54.15 ± 0.72 (3 s) | **48.23 ± 0.43** (3 s; bk16 + 10 k) |
| 32 | 13.77 | 96.85 | bk24 77.45 | bk16 91.26 | 75.23 | **69.27** (bk24 + 10 k) |

## 4.5 A second, independently trained model

To check that the effect is not specific to one training run, we train v7s with the same recipe, data and evaluation exclusions but a narrower width (41.3 M vs 72.5 M parameters), a different seed and 60 k steps; its own step-10 k EMA serves as snapshot. Table 4 shows the same ordering at 16, 20 and 24 px (−62.4 % bare → composed at 16 px for seed 0, −63 % for seed 1), and the FD decomposition mirrors v7h: mean term 16.48 → 5.50 (label) / 7.04 (autoguidance) / 3.84 (composed), covariance term 11.51 → 7.31 / 6.91 / 6.70. A third model, v7_lowres, trained on data that included the evaluation sprites, shows the same relative gain (16.66 → 7.40 and 15.29 → 7.08, seeds 0 / 1) but is contaminated and appears only in the appendix.

**Table 4.** Second model v7s (clean), matched FD-DINOv2. 16 px: seed 0 / seed 1; 20 and 24 px: seed 0. v7h row repeated from Table 1 for reference.

| Model | CFG $w=4$ | Label reference, $w=2$ | Autoguidance 10 k, $w=1.5$ | Composed, $w=1.5$ | Reverse `bucket:20`, $w=2$ | Δ bare → composed |
|---|---|---|---|---|---|---|
| v7h @16 (72.5 M) | 21.98 | bk12 8.59 | 8.98 | **7.67** | 17.14 | −65 % |
| v7s @16 (41.3 M) | 28.00 / 27.75 | bk12 12.80 / 12.74 | 13.95 / 15.17 | **10.54 / 10.28** | 25.31 | −62 % / −63 % |
| v7s @20 | 63.32 | bk16 45.81 | 39.13 | **36.31** | — | −43 % |
| v7s @24 | 99.47 | bk16 78.18 | 66.83 | **60.20** | — | −39 % |

## 4.6 Reverse controls: the reference must be a *lower* bucket

If a wrong label merely acted as a generic unconditional-like reference, a higher bucket would help as much as a lower one. It does not (Table 5, seed 0). At 12, 20 (bk32), 24 and 32 px the higher-bucket reference is *worse than no guidance* (+1.5, +13.2, +21.9, +17.1 FD against the seed-0 bare rows; against the 3-seed means 13.02 ± 0.28 / 47.04 ± 1.11 / 78.96 ± 0.74 the reverse rows 14.45 / 59.13 / 101.67 remain above bare by many sd); at 20 px `bucket:24` is indistinguishable from bare (+0.7). At 16 px higher buckets are slightly better than bare in raw FD but worse after quantisation (q16 18.64 / 21.02 / 24.56 vs 12.64), with precision rising to .935 and recall falling to .85–.88, the signature of over-guidance contraction; v7s shows the same (`bucket:20`: q16 24.14, precision .925, recall .848). `bucketmix:12` (9.42) lies between the label reference (8.59) and CFG (21.98; all seed 0), not at CFG.

**Table 5.** Reverse controls (higher bucket as weak reference, $w=2$), v7h, seed 0.

| $R$ | Bare CFG | Best lower / composed | Higher-bucket reference | Effect vs bare |
|---|---|---|---|---|
| 12 | 12.91 | — / autoguidance 9.18 | bk16 **14.45** | worse (+1.5) |
| 16 | 21.98 | bk12 8.59 / 7.67 | bk20 17.14; bk24 17.27; bk64 19.34 | slightly better raw, worse after q16 |
| 20 | 45.92 | bk16 32.89 / 29.66 | bk24 **46.62**; bk32 **59.13** | ≈ bare (+0.7) / worse (+13.2) |
| 24 | 79.80 | bk16 60.41 / 48.70 | bk32 **101.67** | worse (+21.9) |
| 32 | 96.85 | bk24 77.45 / 69.27 | bk48 **113.93** | worse (+17.1) |

## 4.7 Trained degraded-view branches

Table 6 compares the free label reference with two trained degraded-view branches on the same fine-tuned weights. The block-average branch (`probe_cg`) is *harmful* (21.53 vs bare 19.17 at $w=1.5$; 35.22 and 59.38 at $w=2, 3$); the contrast-shrunk branch (`probe_cc`) helps (14.71 vs 20.63) but stays far behind the label reference on the same weights (9.43) and autoguidance (8.77). At 12 px, where no lower bucket exists, `probe_cc` improves on bare (9.66 vs 11.78) but not on autoguidance (8.16). §5 analyses why.

**Table 6.** Same-weights paired controls inside the trained probes, 16 px, seed 0 (12 px row: no lower bucket).

| Fine-tuned weights | Bare CFG $w=4$ | Trained branch, $w=1.5$ | `bucket:12`, $w=2$ | Autoguidance 10 k, $w=1.5$ |
|---|---|---|---|---|
| `probe_cg` (2×2 block average) | 19.17 (q16 12.37) | 21.53 (q16 16.05) | 10.12 (q16 9.51) | 9.10 (q16 8.45) |
| `probe_cc` (contrast shrink 0.6×) | 20.63 (q16 12.66) | **14.71** (q16 9.17) | 9.43 (q16 8.97) | 8.77 (q16 7.78) |
| `probe_cc` @12 px | 11.78 | 9.66 ($w=1.25$) / 10.07 ($w=1.5$) | — | 8.16 |

## 4.8 Sampler robustness

The composed reference is not an artefact of the 100-step sampler: with 50 / 100 / 200 DDPM steps it gives 6.81 / 7.67 / 9.38 (16 px, seed 0); fewer steps are slightly *better*, consistent with extrapolation error accumulating over steps (the 50-step result awaits a seed check). DDIM-50 is unusable for this model irrespective of guidance (bare 204.42), so DDIM rows appear only in the appendix with that caveat.

## 4.9 Text alignment

Guidance against a same-caption reference removes the unconditional CFG term and therefore weakens the text direction. Table 7 quantifies this with CLIP ViT-B/32 similarity (100·cos) and retrieval R@1 among 1 + 99 captions (chance 1 %) on the *same* saved samples. Every reference-guided row, autoguidance included, sits about 0.5 CLIP below bare CFG at 16 px (R@1 18 → 13–15 %), growing to −0.9 at 32 px. The cost is method-independent (snapshot and label references within 0.2), grows with $w$ (`bucket:12` $w=3$: 29.25), and is absent for reverse references. Bare CFG scores *above* the real sprites (30.06 vs 29.80), i.e. CFG over-aligns, while the guided rows sit ≈ 0.3 below real.

**Table 7.** CLIP ViT-B/32 100·cos (R@1 in parentheses) of the same samples as Tables 1 and 3; seed 0 / seed 1 where two values are given.

| $R$ | Real | CFG $w=4$ | Lower-bucket ref, $w=2$ | Autoguidance, $w=1.5$ | Composed, $w=1.5$ | Higher-bucket ref |
|---|---|---|---|---|---|---|
| 12 | 29.55 (13.9 %) | 29.76 / 29.74 (15 %) | — | 29.41 / 29.39 (13 %) | — | bk16 29.32 (11.8 %) |
| 16 | 29.80 (16.4 %) | 30.06 / 30.04 / 30.03 (18 %) | bk12 29.37 / 29.38 / 29.37 (13 %) | 29.58 / 29.53 / 29.57 (15 %) | 29.51 / 29.47 / 29.54 (14 %) | bk24 29.58; bk64 29.60 (15 %) |
| 20 | 29.86 (19.3 %) | 30.16 / 30.09 (20 %) | bk16 29.34 / 29.35 (14 %) | 29.45 / 29.52 (15–16 %) | 29.48 / 29.49 (15 %) | bk32 29.62 (16 %) |
| 24 | 29.81 (20.2 %) | 30.11 / 30.12 (20 %) | bk16 29.09 / 29.11 (13 %) | 29.35 / 29.38 (15 %) | 29.33 / 29.34 (15 %) | bk32 29.48 (16 %) |
| 32 | 29.65 (20.9 %) | 29.72 (21.5 %) | bk24 28.49 (14 %) | 28.69 (15 %) | 28.78 (15 %) | bk48 28.85 (17.5 %) |

We tested two zero-training fixes (Table 8): (a) `bucketu:12`, a reference under the lower bucket *and* the empty caption, so the guidance direction contains the CFG text direction at the same 2 NFE; (b) an additive plain-CFG term (`cfg_text`) on top of the reference term (3 NFE). None of the seven variants reaches the CLIP of bare CFG; the points lie on an FD–CLIP frontier (29.37 @ 8.59 → 29.77 @ 8.60 → 29.89 @ 11.10) on which every +0.1 CLIP costs roughly +0.5–1 FD, mostly in the mean term. The cheapest operating point is the composed reference under `bucketu:12` at $w=1.5$: FD 8.60 (+0.9, about 3 seed sd) with CLIP at the real-data level (29.77 vs 29.80; R@1 16.1 % vs 16.4 %) at unchanged cost. The main tables keep the un-fixed rows.

**Table 8.** Alignment fixes, 16 px, seed 0.

| Row | NFE | FD | +q16 | CLIP | R@1 |
|---|---|---|---|---|---|
| Bare CFG $w=4$ | 2 | 21.98 | 12.64 | 30.06 | 18.3 % |
| `bucket:12` $w=2$ | 2 | 8.59 | 8.99 | 29.37 | 13.1 % |
| Composed $w=1.5$ | 2 | **7.67** | 8.34 | 29.51 | 14.4 % |
| (a) `bucketu:12` $w=2$ / $w=1.5$ | 2 | 12.10 / 10.45 | 9.24 / 8.60 | 29.84 / 29.77 | 16.1 / 16.4 % |
| (a) composed under `bucketu:12`, $w=1.5$ / $w=1.25$ | 2 | 8.60 / 9.46 | 9.32 / 9.60 | 29.77 / 29.65 | 16.1 / 15.7 % |
| (b) `bucket:12` $w=2$ + `cfg_text` 1.5 | 3 | 9.50 | 8.71 | 29.71 | 16.2 % |
| (b) composed $w=1.5$ + `cfg_text` 1.5 / 2 | 3 | 9.19 / 11.10 | 8.57 / 9.48 | 29.72 / 29.89 | 15.4 / 15.9 % |

## 4.10 A second metric family

Because DINOv2 sees one patch per pixel at these resolutions, we also compute Inception-v3 clean-FID and KID [TODO cite clean-fid] on the same saved samples (white composite, NEAREST ×4 to 64 px). Inception has far less headroom here (floors 4.93 / 6.72 / 8.23 / 8.99 / 9.75 at 12–32 px; bare 9.57 vs floor 6.72 at 16 px). Table 9 shows that the two central claims survive the change of feature space: the composed reference is best or tied-best at every resolution and on both models, and higher-bucket references are harmful. Under Inception, autoguidance and composed are within seed spread at 16 px; the composed advantage is clear at 20 / 24 / 32 px. The *label-only* reference, however, is DINOv2-visible but Inception-weak: it lowers FID at 16 / 20 px but is flat or slightly worse at 24 / 32 px (16.13 → 16.45, KID 2.40 → 3.19) and lags autoguidance on v7s (9.77 vs 8.25), in line with the q16 result (Appendix, Table (g′)): the label reference mainly corrects per-pixel colour and contrast statistics, which DINOv2 weights heavily and 64 px Inception barely sees. Rank agreement between the metric families over all saved rows is Spearman .98 / .86 / .96 / .68 / .83 at 12 / 16 / 20 / 24 / 32 px (Pearson .91–.99).

**Table 9.** Inception-v3 clean-FID (KID ×10⁻³ where given), same samples as Tables 1, 3, 4. 16 px FID: seeds 0 / 1 / 2.

| $R$ | Bare CFG | Lower-bucket ref, $w=2$ | Autoguidance 10 k | Composed | Reverse (higher bucket) |
|---|---|---|---|---|---|
| 16, FID (3 seeds) | 9.57 / 9.61 / — | bk12 8.61 / 8.63 / 8.64 | **7.84** / 7.97 / 7.71 | 7.97 / 7.86 / 7.95 | bk24 11.26; bk64 11.29 |
| 16, KID | 1.12 | .78 | .55 | .63 | 2.71 |
| 20, FID / KID | 11.82 / 1.01 | bk16 10.75 / 1.03 | 9.85 / .58 | **9.62 / .26** | bk32 15.45 / 3.71 |
| 24, FID / KID | 16.13 / 2.40 | bk16 16.45 / 3.19 (no gain) | 13.83 / 1.96 | **13.26 / 1.46** | bk32 22.56 / 7.55 |
| 32, FID / KID | 19.77 / 2.34 | bk24 20.10 / 3.31 (no gain) | 18.56 / 2.91 | **17.24 / 1.81** | bk48 25.83 / 7.58 |
| v7s @16, FID | 10.33 | bk12 9.77 | **8.25** | 8.49 | bk20 12.86 |
| v7s @20 / @24, FID | 13.06 / 16.82 | 11.73 / 17.85 | 10.42 / 14.66 | **10.22 / 13.85** | — |

## 4.11 Cost

Table 10 gives wall-clock and peak memory for 1,000 samples at 16 px on one A100-80GB (batch 500, 100 DDPM steps, model load included). The label reference costs exactly what CFG costs; snapshot-based references hold one extra copy of the weights (+556 MiB) at the same wall-clock; the 3-NFE alignment variant is +43 %.

**Table 10.** Cost per 1,000 samples @16 px, v7h (72.5 M).

| Configuration | NFE / step | s / 1,000 samples | Peak memory (MiB) |
|---|---|---|---|
| No guidance ($w=1$) | 1 | 42.5 | 4951 |
| CFG $w=4$ | 2 | 73.8 | 4951 |
| Label reference `bucket:12`, $w=2$ | 2 | 72.7 | 4951 |
| Autoguidance (snapshot 10 k), $w=1.5$ | 2 | 73.3 | 5507 |
| Composed, $w=1.5$ | 2 | 74.0 | 5507 |
| Composed + `cfg_text` 1.5 | 3 | 105.9 | 5507 |

## 4.12 Qualitative results

Figures `fig_qual_12px.png`, `fig_qual_16px.png`, `fig_qual_20px.png`, `fig_qual_24px.png` and `fig_qual_32px.png` show prompt-aligned samples at each resolution (16 px: 6 rows × 24 sprites, same seed and prompt per column; rows include bare CFG, autoguidance, `bucket:12` and composed). The bare model's characteristic failures are bleeding colours and insufficient local contrast, which the guided rows remove [TBD: confirm wording against the final figure]. Guided samples still have roughly 1.8× the colour count and one quarter of the flat-region fraction of real sprites, which accounts for the remaining gap to the floor (7.5 vs 3.45) analysed in §5.
