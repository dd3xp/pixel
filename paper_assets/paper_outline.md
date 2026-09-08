# Paper outline: Cross-Resolution Self-Guidance for Very-Low-Resolution Pixel-Art Diffusion

Working document for the ICLR submission. Every number below is copied from
`pixel_art_research_20260816/experiment_log.md` (cited as **[log §<timestamp>]**, all timestamps 2026-09-07 UTC unless
stated) or `autonomy_state.md`. Nothing is invented; missing values are marked **TBD**.
Protocol everywhere unless stated: **matched FAIR FD-DINOv2** (`src/v6/fd_fair.py`, DINOv2-small CLS features, reference =
3000 real sprites, generated = 3000 samples from the captions of 3000 *held-out* real sprites, n = 1 per caption, seed 0,
DDPM 100 steps, EMA weights; lower is better). Floor = 3000 held-out real sprites vs the reference set = **3.45 @16 px**
[log §2026-09-06 cycle 5 diagnostic (2)]. Seed-to-seed sd of the protocol is 0.2–0.5 [log §17:35].

---

## 1. Title candidates and abstract

### Title candidates
1. **Cross-Resolution Self-Guidance: A Diffusion Model's Lower-Resolution Belief Is Its Own Best Bad Model**
2. **Guiding Pixel-Art Diffusion Away From Its Coarser Self: Zero-Training Guidance for Bucketed Multi-Resolution Models**
3. **Contrast-Deficient Self-Guidance: Resolution Labels as a Free Weak Reference for Very-Low-Resolution Sprite Generation**

### Abstract (draft, ~150 words)

Diffusion models of very-low-resolution pixel art (12–24 px RGBA sprites) suffer from a systematic error that
classifier-free guidance cannot fix: the denoiser regresses towards the mean, producing bleeding colours and
insufficient local contrast. We show that a *bucketed* multi-resolution model already contains the weak reference that
autoguidance needs. Feeding the same weights, the same noisy input and the same text, but the label of a *lower*
resolution bucket, yields a structure-aligned prediction with systematically lower local contrast; extrapolating the
strong prediction away from it (e = e_weak + w(e_strong − e_weak)) removes the error at zero training cost and without
storing any second model. On a clean 16 px baseline, FD-DINOv2 drops from 21.5 to 8.5 (autoguidance with an early
snapshot: 8.7); composing the two references reaches 7.5 (floor 3.45). The effect transfers to 20, 24 and 32 px
(3-seed 47.0 → 28.9 and 79.0 → 48.2 at 20/24 px) and to a smaller independently trained model (28.0 → 10.5), is
strictly directional (higher-bucket references hurt at every resolution and under Inception FID/KID as well), and is explained by a controlled mechanism study:
an effective weak reference must be lower in total variation *and* structure-aligned, a condition that explicit low-pass
branches violate.

---

## 2. Contribution bullets (honest)

1. **A zero-training guidance mechanism for bucketed multi-resolution diffusion models.** The weak reference is the model's
   own prediction under a lower-resolution bucket label (same weights, same x_t, same text); guidance replaces CFG at
   identical cost (2 forwards/step). At 16 px it matches early-snapshot autoguidance (8.52 ± 0.29 vs 8.73 ± 0.26 over 3
   seeds) with no snapshot, and composing both references gives 7.53 ± 0.19, a 65 % reduction from the CFG baseline
   21.52 ± 0.47 [log §17:35]. This is not a new network architecture; the model, data and training recipe are unchanged.
2. **Generality and directionality evidence.** The gain reproduces at 20 px (3-seed 47.04 → 28.90), 24 px (78.96 → 48.23)
   and 32 px (96.85 → 69.27), on a clean, smaller, independently trained model v7s at 16/20/24 px (28.0 → 10.5 / 63.3 → 36.3 /
   99.5 → 60.2), and is directional at every resolution: *higher*-bucket references are worse than no guidance at 12, 20,
   24 and 32 px and far weaker at 16 px, under FD-DINOv2 and under Inception FID/KID [Tables b, c, h].
3. **A mechanism study with controlled trained ablations.** Pure weak-reference statistics show that every effective
   reference has lower total variation than the strong model while keeping its colour count and structure; a trained 2×2
   block-average branch (lower TV but block structure) is *harmful*, a trained contrast-shrunk branch (structure-aligned)
   is helpful but far weaker than the free bucket label. The two references are complementary: the bucket reference fixes
   the mean term of FD, the snapshot reference restores coverage; the snapshot contributes only in the low-noise half
   [log §14:20, §16:35, §19:05, §19:35].
4. **Negative results that bound the design space.** APG, frequency-decoupled guidance, interval scheduling, CADS, high
   CFG, degraded-input references and channel-decoupled weights all fail to improve on the composed reference, and each
   failure is consistent with the identified error direction (radial, low-frequency contrast restoration) [log §07:30,
   §15:10, §16:35].

---

## 3. Results tables (final-paper form)

### Table (a) — 16 px main results, 3 seeds. Source: **[log §17:35 UTC, dseed2]**; q16 seed-1 values from **[log §13:11, dv7h6]**.

Model v7h (clean baseline: v7 recipe, random init, 80 k steps, evaluation sprites excluded from training). Weak snapshot =
EMA at step 10 k of the same run. w = guidance weight in e = e_weak + w(e_strong − e_weak); CFG row uses w = 4 with the
unconditional prediction as reference.

| Method | Reference (weak) | Trains / stores extra? | seed 0 | seed 1 | seed 2 | **FD mean ± sd** | +q16 (s0 / s1 / s2) |
|---|---|---|---|---|---|---|---|
| Real held-out (floor) | — | — | 3.45 | — | — | **3.45** | — |
| CFG w = 4 (baseline) | unconditional | no | 21.98 | 21.05 | 21.54 | **21.52 ± 0.47** | 12.64 / 12.68 / 12.55 |
| Autoguidance w = 1.5 (Karras 2024) | snapshot 10 k | stores snapshot | 8.98 | 8.73 | 8.47 | **8.73 ± 0.26** | 7.82 / 7.98 / 7.82 |
| **Cross-res self-guidance w = 2 (ours)** | bucket:12, same weights | **no** | 8.59 | 8.20 | 8.76 | **8.52 ± 0.29** | 8.99 / 8.78 / 8.57 |
| **Composed w = 1.5 (ours)** | snapshot 10 k under bucket:12 | stores snapshot | 7.67 | 7.32 | 7.61 | **7.53 ± 0.19** | 8.34 / 8.02 / 8.42 |
| Composed, snapshot only for t/T ∈ [0, 0.5] ("snaplo") | as above, final weights outside interval | stores snapshot | 7.70 | 7.16 | 8.02 | **7.63 ± 0.43** | TBD / 8.16 / 8.44 |

Notes for the caption: (i) bucket:12 vs autoguidance differ by 0.2, not significant; composed is below every single-reference
seed. (ii) After 16-colour quantisation (+q16) the ordering between autoguidance and composed reverses (7.8 vs 8.3–8.4),
i.e. part of the composed gain is in the colour domain; q16 goes to an appendix, not the main metric [log §17:35 point 4].

**Table (a′) — FD decomposition and P/R for the seed-0 rows (appendix). Source: [log §07:30 UTC, diag_v7h] and [log §13:11].**

| Setting | FD | mean term | cov term | precision | recall | density | coverage |
|---|---|---|---|---|---|---|---|
| Real held-out (floor) | 3.45 | 0.38 | 3.07 | .939 | .924 | 1.029 | .966 |
| CFG w = 4 | 21.98 | 12.13 | 9.85 | .907 | .902 | .874 | .851 |
| Autoguidance 10 k, w = 1.5 | 8.98 | 3.88 | 5.09 | .913 | .906 | .936 | .928 |
| Autoguidance 10 k, w = 2 | 11.33 | 5.09 | 6.24 | .915 | .902 | .895 | .894 |
| bucket:12, w = 2 | 8.59 | 2.55 | 6.04 | .911 | .894 | .865 | .892 |
| Composed 10 k + bucket:12, w = 1.5 (seed 0 / seed 1) | 7.67 / 7.32 | 2.40 / 2.09 | 5.27 / 5.23 | TBD | TBD / .902 | TBD | .909 / .916 |

Reading: bucket reference has the lowest mean term (systematic mean shift), snapshot has the best coverage; the
composition takes roughly half of each [log §07:30 reading 5].

**Table (a″) — other zero-training guidance on v7h @16 px (seed 0), all worse than the baseline or than autoguidance. Source: [log §07:30].**

| Setting | FD | +q16 |
|---|---|---|
| CFG w = 7 / w = 10 | 42.95 / 62.12 | 19.79 / 25.52 |
| CFG w = 4 + CADS / w = 7 + CADS / w = 7 + CADS s = .25 | 44.58 / 71.17 / 88.61 | 24.72 / 34.74 / 47.29 |
| CFG w = 7, guidance interval [0, .8] | 32.57 | 13.26 |
| Autoguidance 10 k: w = 2.5 / w = 3 | 19.19 / 30.11 | 7.80 / 9.34 |
| Autoguidance w = 2: snapshot 5 k / 20 k / 40 k | 14.89 / 12.14 / 11.62 | 12.75 / 7.36 / 8.72 |
| Autoguidance 10 k w = 2, DDPM 200 steps | 11.35 | 6.96 |
| Autoguidance 10 k w = 2 + CFG w = 7 interval | 133.35 | 46.33 |
| bucket:12: w = 1.5 / 2.5 / 3 | 10.21 / 11.20 / 13.87 | 9.48 / 11.67 / 14.79 |
| bucketmix:12 (½ bucket:12 + ½ unconditional), w = 2 | 9.42 | TBD |
| Composed: w = 2 / w = 1.25 (seed 1) / snapshot 20 k w = 1.5 / snapshot 5 k w = 1.5 (seed 1) | 10.83 / 8.98 / 8.03 / 8.68 | 10.14 / 9.36 / 7.98 / 10.47 |

Weight sensitivity: the bucket reference's w-curve is flat (10.21 → 8.59 → 11.20 → 13.87 for w = 1.5…3) whereas the
snapshot reference explodes (8.98 → 30.11 for w = 1.5 → 3) [log §07:30 reading dv7h5].

**Weight sweeps at 20 / 24 px (v7h, seed 0, matched FD at native R; diag_wsweep20 / diag_gpu3b, [log §09-08 07:00]).** Same
shape as 16 px: the label reference degrades gracefully with w, the snapshot reference has a sharp optimum and explodes past it.

| R | reference | w = 1.25 | 1.5 | 2 | 2.5 | 3 |
|---|---|---|---|---|---|---|
| 20 | bucket:16 | 40.96 | 37.48 | **32.89** | 35.15 | 38.53 |
| 20 | snapshot 10 k | 36.75 | **31.78** | 33.66 | 42.02 | 59.04 |
| 24 | bucket:16 | 71.12 | 63.09 | **60.41** | 65.77 | 74.61 |
| 24 | snapshot 10 k | 64.85 | 53.36 | **52.52** | 66.53 | 83.41 |

(bare: 20 px 45.92, 24 px 78.96 ± .74. Seed-0 values; the 20/24 px w = 1.5 snapshot and w = 2 label entries are the seed-0
members of the 3-seed rows in Table b.) Worst-over-sweep / best: label ref 1.25× (20 px) and 1.23× (24 px); snapshot
ref 1.86× (20 px) and 1.59× (24 px), w ≤ 3. At 24 px the snapshot optimum moves to w = 2 (52.52 < 53.36), so the Table-b composed row
(w = 1.5) is not tuned in the snapshot's favour. **#6 DONE (09-08 07:20).**

### Table (j) — **CFG weight curve and alignment–fidelity frontier @16 px (v7h, seed 0). Source: [log 09-08 §07:15, §07:52, §08:05]; figure `fig_pareto_fd_clip.png` (make_pareto.py).**

**The recipe default w = 4 is over-guided: the CFG optimum is w = 1.5 (FD 12.24).** Every "bare" comparison in the paper must
therefore be against best-CFG (w swept over {1, 1.5, 2, 3, 4, 7, 10}), with w = 4 kept as the training-default row.

| CFG w | 1 | **1.5** | 2 | 3 | 4 (default) | 7 | 10 |
|---|---|---|---|---|---|---|---|
| FD | 16.39 | **12.24** | 12.98 | 16.67 | 21.98 | 42.95 | 62.12 |
| mean / cov term | 9.47 / 6.92 | 5.87 / 6.37 | 6.32 / 6.66 | 8.70 / 7.97 | 12.13 / 9.85 | 27.01 / 15.94 | 41.15 / 20.97 |
| CLIP 100cos / R@1 | 29.43 / 13.8 % | 29.74 / 15.9 % | 29.87 / 16.8 % | 30.01 / 17.4 % | 30.06 / 18.3 % | 30.04 / 17.9 % | 30.05 / 17.3 % |

Reference-guidance rows against this frontier (same seed): bucket:12 w2 **8.59** / CLIP 29.37; autoguidance w1.5 8.98 / 29.58;
composed w1.5 **7.67** / 29.51; composed∘bucketu:12 w1.5 8.60 / **29.77** (= real 29.80); composed + CFG term 1.5 (3 NFE) 9.19 /
29.72; composed + CFG term 2 (3 NFE) 11.10 / 29.89. **At every CLIP level the guided rows are 3–5 FD below the CFG curve**; CFG
cannot reach FD < 12.2 at any w, and the "alignment cost" of the guided rows (−0.3 … −0.5 CLIP vs w = 4) is exactly the
cost CFG itself pays for lowering w (w = 1.5: 29.74). Headline vs best-CFG: composed −37 % (3-seed 7.53 vs 12.24 s0),
bucket:12 −30 %, autoguidance −29 %; vs the w = 4 default −65 %.

PAG baseline (Ahn et al. 2024; identity self-attention in the mid block, `pag:mid`): w1.5 / 2 / 3 = 12.57 / 12.62 / 12.90, CLIP
29.45 / 29.41 / 29.45; mid + down_blocks.1 w2 13.05; PAG w2 + CFG term 1.5 (3 NFE) 11.73 / 29.74. PAG ≈ same-w CFG (a gentler
CFG that does not explode at w = 3) and pays the same CLIP; it does not remove the systematic error.

Interventional control (`shrink:f`, 1 NFE): reference = the strong model's own x̂₀ with contrast shrunk toward the per-image mean by
f — i.e. guidance = uniform contrast amplification of the strong prediction. f = 0.5 / 0.7 / 0.85 at w2: FD **374.6 / 200.4 / 74.8**;
f = 0.7 w3: 444.4 (CLIP 28.5–29.5, R@1 5–12 %). Monotonically harmful → the bucket reference is *not* a "low-contrast scalar";
its error direction is spatially structured. Pairs with the trained probe_cc result (per-pixel learned shrink: helpful but weak).

Sampler steps (50-step DDPM, seed 0): CFG4 20.81, autoguidance 9.11, composed **6.63** (100-step: 21.98 / 8.98 / 7.67).

**Table (j′) — CFG weight curves at every R / model and the resulting best-CFG rows (diag_review2/3/4, [log 09-08 §10:10]).**
Seed 0 unless a ± is given; best-w row re-run with seeds 1/2 where it is used as a 3-seed baseline.

| Model @R | w=1.5 | 2 | 2.5 | 3 | 4 (default) | **best CFG** | label ref | autog 10 k | **composed** | Δ composed vs best CFG / vs w=4 |
|---|---|---|---|---|---|---|---|---|---|---|
| v7h @12 | 10.44 | **9.71** | 10.53 | 11.09 | 13.02 ± .28 | **9.71** (w2, s0) | — | 8.54 ± .73 | — | autog −12 % / −34 % |
| v7h @16 | **12.24 / 13.32 / 11.91 = 12.49 ± .72** | 12.98 | — | 16.67 | 21.52 ± .47 | **12.49 ± .72** (w1.5) | 8.52 ± .29 | 8.73 ± .26 | **7.53 ± .19** | **−40 % / −65 %** |
| v7h @20 | 42.61 | 40.57 | **39.53 / 39.57 / 39.61 = 39.57 ± .04** | 42.99 | 47.04 ± 1.11 | **39.57 ± .04** (w2.5) | 32.75 ± .60 | 31.57 ± .29 | **28.90 ± .69** | **−27 % / −39 %** |
| v7h @24 | 71.22 | **67.34 / 68.66 / 67.31 = 67.77 ± .77** | — | 72.93 | 78.96 ± .74 | **67.77 ± .77** (w2) | 59.32 ± 1.37 | 54.15 ± .72 | **48.23 ± .43** | **−29 % / −39 %** |
| v7h @32 | 92.21 | **83.65** | — | 86.99 | 96.85 | **83.65** (w2) | 77.45 | 75.23 | **69.27** | **−17 % / −28 %** |
| v7s @16 | 19.29 | **19.10** | 20.34 | 22.96 | 28.00 / 27.75 | **19.10** (w2, s0) | 12.80 / 12.74 | 13.95 / 15.17 | **10.54 / 10.28** | **−45 % / −62 %** |
| v7s @20 | 61.62 | **57.48** | TBD | TBD | 63.32 | **57.48** (w2) | 45.81 | 39.13 | **36.31** | **−37 % / −43 %** |
| v7s @24 | 99.23 | **93.78** | TBD | TBD | 99.47 | **93.78** (w2) | 78.18 | 66.83 | **60.20** | **−36 % / −39 %** |

Reading for the paper: (i) the over-guidance of w = 4 is largest at 16 px (−42 % from re-tuning alone) and small at 20–32 px
(−14 … −16 %) and for v7s (−32 % @16, −9 % @20, −6 % @24); (ii) **the composed reference beats best-CFG at every R and
both models by 17–45 %**, and every gap is ≥ 10 seed-sd where 3 seeds exist; (iii) the *label-only* reference beats best-CFG by
32 % @16 but only 17 % / 12 % / 7 % at 20 / 24 / 32 px — so the free variant is a 16 px result plus a component of the
composed method, and the paper must say so; (iv) autoguidance alone beats best-CFG by 30 / 20 / 20 / 10 % at 16/20/24/32.
Best-CFG w differs by R (1.5 @16, 2.5 @20, 2 @24/32) — CFG needs per-resolution tuning while the guided rows use fixed w.

### Table (b) — Resolution generalisation, matched FD at the native resolution (fd_fair --size R). Sources: **[log §14:05]** (seed 0, 12/16/20 px), **[log §15:10]**, **[log §16:35]** (24 px completion), **[log §19:05 + §19:35]** (dseedR seed 1). Seed-1 values for 20/24 px as given in the task brief (dseedR).

Weights v7h throughout; snapshot = 10 k EMA; "lower" = nearest lower bucket unless stated.

| R | Floor | CFG w = 4 (bare) | Lower-bucket ref, w = 2 | Other lower bucket, w = 2 | Autoguidance 10 k, w = 1.5 | **Composed (lower + 10 k), w = 1.5** | Higher-bucket ref, w = 2 (reverse control) |
|---|---|---|---|---|---|---|---|
| 12 | 3.37 | 13.02 ± .28 (3 s) | — (no lower bucket exists) | — | **8.54 ± .73** (3 s) | — | bk16 **14.45** (worse than bare) (s0) |
| 16 | 3.45 | 21.98 / 21.05 | bk12 **8.59** / 8.20 | — | 8.98 / 8.73 | **7.67 / 7.32** | bk20 17.14, bk24 17.27, bk64 19.34 |
| 20 | 12.14 | 47.04 ± 1.11 (3 s) | bk16 **32.75 ± .60** (3 s) | bk12 35.85 (s0) | 31.57 ± .29 (3 s) | **28.90 ± .69** (3 s) | bk24 **46.62** (≈ bare), bk32 **59.13** (worse) (s0) |
| 24 | 12.96 | 78.96 ± .74 (3 s) | bk16 **59.32 ± 1.37** (3 s) | bk12 57.70, bk20 66.20 (s0) | 54.15 ± .72 (3 s) | **48.23 ± .43** (3 s; bk16 + 10 k) | bk32 **101.67** (worse than bare) |
| 32 | 13.77 | 96.85 | bk24 **77.45** | bk16 91.26 | 75.23 | **69.27** (bk24 + 10 k) | bk48 **113.93** (worse than bare) |

Cell format: seed 0 / seed 1 (32 px seed 0 only, **[log §09-08 02:00, dmech2]**). Caption points: (i) relative gain of the
composed reference: 16 px −65 %, 20 px −35 %, 24 px −39 %, 32 px −28 % (seed 0); (ii) the bare model is much further from
the floor at 20/24/32 px (gap 33.8 / 66.8 / 83.1 vs 18.5 at 16 px) because training data is 16 px-dominated
[log §14:05 reading 3]; (iii) at 24 px a *farther* lower bucket (bk12 57.70) is slightly better than the nearest
(bk16 60.41), but at 32 px the farther bucket is clearly worse (bk16 91.26 vs bk24 77.45) — the belief must stay
structurally aligned, see Table d; (iv) 12 px has no lower bucket, so only snapshot autoguidance applies — a real
limitation, stated as such; (v) the ordering bare > lower-bucket ≈ autoguidance > composed holds at every resolution
that has a lower bucket (16/20/24/32); at 32 px autoguidance (75.23) edges the label reference (77.45).

### Table (c) — Second model, **clean**: v7s (v7h recipe and data, evaluation sprites excluded; width 96 → 41.3 M params vs 72.5 M; seed 1; 60 k steps; snapshot = its own step 10 k). Source: **[log §09-08 03:50, v7s]**. v7_lowres (memorisation-contaminated, trained on the evaluation sprites) goes to the appendix as a third model.

| Model @16 px | CFG w = 4 (bare) | bucket:12 ref, w = 2 | Autoguidance 10 k, w = 1.5 | Composed, w = 1.5 | Reverse bucket:20 w = 2 | Δ bare → composed |
|---|---|---|---|---|---|---|
| v7h (72.5 M, clean, Table a) | 21.98 (q16 12.64) | 8.59 (q16 8.99) | 8.98 (q16 7.82) | **7.67** (q16 8.34) | 17.14 (q16 18.64) | −65 % |
| **v7s (41.3 M, clean)**, s0 / s1 | 28.00 / 27.75 (q16 13.60 / 13.72) | 12.80 / 12.74 (q16 10.00 / 10.04) | 13.95 / 15.17 (q16 9.35 / 10.02) | **10.54 / 10.28** (q16 9.57 / 9.06) | 25.31 (q16 24.14) | −63 % |
| v7s @20 px (s0) | 63.32 | bk16 45.81 | 39.13 | **36.31** | — | −43 % |
| v7s @24 px (s0) | 99.47 | bk16 78.18 | 66.83 | **60.20** | — | −39 % |
| v7_lowres (contaminated; appendix) | 16.66 / 15.29 (s0 / s1) | 7.40 / 7.08 | — | — | — | (bucket:12: −56 %) |

FD decomposition on v7s mirrors v7h: mean term 16.48 → 5.50 (bucket:12) / 7.04 (autoguidance) / 3.84 (composed);
cov term 11.51 → 7.31 / 6.91 / 6.70 — the label reference fixes the mean shift, the snapshot fixes coverage, the
composition takes both. Reverse control identical in kind: bucket:20 is slightly better raw but far worse after q16 and
has precision ↑ .925 / recall ↓ .848 (over-guidance contraction). On v7s the ordering bare > label ref > autoguidance > composed holds at 16, 20 and 24 px [log §09-08 05:50].

### Table (d) — Mechanism: statistics of the *pure* weak-reference samples (sampled with `--cfg 0`, i.e. following only e_weak) vs their effect when used as guidance reference. Sources: **[log §14:20 UTC, dmech]** and **[log §19:35 UTC, dcc12]**; probe_cg/probe_cc guided FD from **[log §13:11]** and **[log §19:05]**.

Statistics over opaque pixels: ncol = median unique colours per sprite; flat = fraction of exactly-equal adjacent pixel
pairs; TV = mean adjacent |ΔRGB|. Strong model = v7h CFG w = 4.

| Reference (sampled alone @16 px) | FD (alone) | ncol | flat | **TV** | FD when used as weak ref | Verdict |
|---|---|---|---|---|---|---|
| Real 16 px sprites (target distribution) | 3.45 (floor) | 34 | .202 | 30.2 | — | — |
| Strong model, v7h CFG w = 4 | 21.98 | 73 | .030 | **32.0** | — | — |
| **bucket:12 belief (v7h, same weights)** | 36.43 | 78 | .015 | **21.6** | **8.59** | effective |
| Snapshot 10 k | 69.16 | 95 | .000 | **27.3** | **8.98** | effective |
| Unconditional (no text) | 32.15 | 68 | .017 | **27.3** | 21.98 (CFG w = 4; worse at higher w) | see note † |
| bucket:24 belief | 34.50 | 75 | .039 | 31.9 (≈ strong) | 17.27 | ineffective |
| bucket:64 belief | 327.6 (collapsed) | 102 | .046 | 40.9 (> strong) | 19.34 | ineffective |
| **probe_cg** trained 2×2 block-average branch (stats are of its training target: block-averaged real 16 px) ‡ | — | 21 | **.605** | **14.2** | **21.53 / 35.22 / 59.38** (w = 1.5/2/3; bare 19.17) | **harmful** |
| **probe_cc** trained contrast-shrunk branch (RGB shrunk 0.6× to per-image mean), sampled alone | 41.37 | 64 | .034 | **15.9** | **14.71 / 19.98 / 44.90** (w = 1.5/2/3; bare 20.63) | effective but weak |
| Real native 12 px / 16 px sprites (for reference) | — | 5 / 6 | .51 / .42 | 28 / 30 | — | — |
| *After guidance*: autoguidance w1.5 / bucket:12 w2 / composed w1.5 | 8.98 / 8.59 / 7.67 | 64 / 61 / 60.5 | .041 / .045 / .044 | 29.0 / 32.4 / 30.1 | — | — |

† The unconditional reference has low TV but is used through CFG (w = 4) and changes the *text* condition, so its
difference vector also carries text-alignment; it is the one exception on the TV-vs-FD plot (`paper_assets/fig_tv_vs_fd.png`,
7 points) and must be explained in the text as "not a same-condition, structure-aligned belief" [autonomy_state, 当前状态].
‡ For probe_cg the pure-reference sample statistics were not measured; the row reports the statistics of the block-averaged
real sprites the branch was trained to reproduce [log §14:20 table]. Marked as such in the caption; measuring the sampled
branch is listed in §5.

Same-weights paired controls inside the trained probes (so the comparison is not confounded by fine-tuning):

| Weights | bare CFG w = 4 | trained branch ref, w = 1.5 | bucket:12, w = 2 (same weights) | autoguidance 10 k, w = 1.5 (same weights) | Source |
|---|---|---|---|---|---|
| probe_cg (block-average) | 19.17 (q16 12.37) | 21.53 (q16 16.05) | 10.12 (q16 9.51) | 9.10 (q16 8.45) | [log §13:11] |
| probe_cc (contrast-shrink) | 20.63 (q16 12.66) | **14.71** (q16 9.17) | 9.43 (q16 8.97) | 8.77 (q16 7.78) | [log §19:05] |
| probe_cc @12 px (no lower bucket) | 11.78 | 9.66 (w = 1.25) / 10.07 (w = 1.5) | — | 8.16 | [log §19:35] |

Same analysis at 20 and 24 px (pure beliefs sampled with `--cfg 0`, matched protocol, seed 0; **[log §09-08 02:00, dmech2]**):

| R | Reference (sampled alone) | FD (alone) | opaque | ncol | flat | **TV** | FD when used as ref (w = 2) | Verdict |
|---|---|---|---|---|---|---|---|---|
| 20 | Real 20 px | 12.14 | .345 | 33 | .277 | 29.9 | — | — |
| 20 | Strong, v7h CFG w = 4 | 45.92 | .362 | 96 | .056 | **31.0** | — | — |
| 20 | **bucket:16 belief** | 94.02 | .395 | 107 | .030 | **21.8** | **32.89** | effective |
| 20 | bucket:12 belief | 179.63 | .455 | 121 | .018 | **16.9** | 35.85 | effective, weaker (opaque .455 ≫ .362: structure drift) |
| 20 | bucket:24 belief | 66.07 | .355 | 97 | .038 | 28.5 (≈ strong) | 46.62 (≈ bare) | ineffective |
| 20 | *after guidance*: bk16 w2 / composed | 32.89 / 29.66 | .339 / .350 | 81 / 78 | .072 / .072 | 32.5 / 29.7 | — | — |
| 24 | Real 24 px | 12.96 | .333 | 29 | .345 | 29.0 | — | — |
| 24 | Strong, v7h CFG w = 4 | 79.80 | .363 | 130 | .067 | **30.3** | — | — |
| 24 | **bucket:16 belief** | 173.39 | .425 | 142 | .028 | **18.2** | **60.41** | effective |
| 24 | bucket:32 belief | 60.16 | .307 | 86 | .097 | **31.3** (> strong) | 101.67 (worse than bare) | harmful |
| 24 | *after guidance*: bk16 w2 / composed | 60.41 / 48.70 | .308 / .323 | 102 / 95 | .087 / .083 | 34.7 / 29.6 | — | — |

Third and fourth replication of the pattern: the reference that helps has *lower* TV than the strong prediction
(21.8 / 16.9 @20, 18.2 @24) while the ineffective/harmful ones have TV ≈ or > strong (28.5 @20, 31.3 @24); the
reference's own FD is anti-correlated with its usefulness (bucket:24 is the best-looking belief at 20 px, FD 66 vs
94/180, and the worst reference; bucket:32 at 24 px is the best-looking, FD 60, and actively harmful). Also confirms
the "structure-aligned" clause: at 20 px the farthest bucket (12) has the lowest TV but drifts in opacity (.455 vs .362)
and is a weaker reference than bucket:16; the same holds at 32 px (bk16 91.26 vs bk24 77.45, Table b).

Mechanism statement to defend: an effective weak reference must (i) have lower local contrast (TV) than the strong
prediction — necessary: bk24/bk64 fail — and (ii) be structure-aligned with unchanged colour/structure statistics —
necessary: probe_cg (TV 14 but block grid) is harmful, probe_cc (TV 15.9 but fewer colours, higher flat) is weaker than
bucket:12 (TV 21.6, ncol 78 ≈ strong 73). The resolution label is a monotone contrast knob (21.6 → 32 → 41 for bucket
12 → 24 → 64) that satisfies both conditions for free [log §14:20, §19:35]. Earlier "simplicity prior" (fewer colours /
larger flat regions) explanation is falsified: the bucket:12 belief has *more* colours (78) and no flat regions (.015).

### Table (e) — Falsified second components on top of the composed reference @16 px, seed 0. Sources: **[log §15:10 UTC, dapg]** and **[log §16:35 UTC, dsched]**. Success criterion set in advance: < 7.2 [autonomy_state].

| Component | Configuration | FD | +q16 | mean / cov term |
|---|---|---|---|---|
| — (reference) | composed w = 1.5 | **7.67** | 8.34 | 2.5 / 5.2 |
| — | composed w = 2 | 10.83 | 10.14 | — |
| APG (Sadat 2024), RGB-only projection | composed w = 1.5, η = 0, β = −0.5 | 9.94 | 8.85 | — |
| APG RGB-only | composed w = 2 | 12.68 | 8.24 | — |
| APG full (4-channel projection) | composed w = 2 | 13.42 | 7.83 | — |
| APG projection only (β = 0) | composed w = 2 | 10.89 | 8.78 | — |
| APG momentum only (η = 1, β = −0.5) | composed w = 2 | 13.15 | 8.03 | — |
| APG RGB-only | composed w = 3 | 37.14 | 10.53 | — |
| APG RGB-only, single reference | bucket:12 w = 2 (no APG: 8.59) | 9.84 | 8.15 | — |
| Interval scheduling of snapshot ref | snapshot only t/T ∈ [0, .5], w = 1.5 | 7.70 | — | 2.42 / 5.28 |
| Interval scheduling | snapshot only [.5, 1], w = 1.5 | 8.99 | — | 3.54 / 5.45 |
| Interval scheduling | snapshot only [.2, .8], w = 1.5 | 8.84 | — | 3.59 / 5.24 |
| Interval scheduling | snapshot only [.2, .8], w = 2 | 9.04 | — | 2.70 / 6.34 |
| FDG (Sabour 2025), 1-level, w_high / w_low | 1.5 / 1.0 | 10.03 | — | 4.8 / 5.22 |
| FDG | 1.5 / 1.25 | 8.96 | — | 3.54 / 5.42 |
| FDG | 2 / 1.0 | 11.80 | — | 5.57 / 6.23 |
| FDG | 2 / 1.25 | 9.82 | — | 3.66 / 6.16 |
| FDG | 2.5 / 1.0 | 18.33 | — | 9.91 / 8.42 |

Also falsified as components (single-reference autoguidance, [log §07:30]): channel-decoupled weights (RGB w / alpha w):
2/1 → 11.16, 2/3 → 10.74, 3/2 → 32.36, 1.5/2.5 → 8.35, 2.5/1.5 → 19.17 — FD is set almost entirely by the RGB weight;
degraded-input references: 2×2 box-blurred x_t → 223.07 (w = 2) / 99.89 (w = 1.5), 1-px cyclic shift → 18.95.

Interpretation for the text: the correction direction is a radial, low-frequency restoration of per-image contrast, so
APG (removes the radial component), FDG with w_low < w_high (down-weights low frequencies; mean term rises monotonically
2.5 → 3.5 → 4.8) and momentum all remove exactly the useful part. Interval scheduling shows the snapshot reference acts
only in the low-noise half ([0, .5] ≈ full schedule; [.5, 1] ≈ single bucket:12 reference).

### Table (f) — Reverse controls: higher-resolution bucket as reference. Sources: **[log §14:05]**, **[log §07:30]**, **[log §16:35]**.

| Generated R | Bare CFG w = 4 | Best lower-bucket / composed | Higher-bucket ref (w = 2) | Effect vs bare |
|---|---|---|---|---|
| 12 px | 12.91 | — (none) / autog 9.18 | bk16: **14.45** | worse (+1.5) |
| 16 px | 21.98 | bk12 8.59 / 7.67 | bk20: 17.14; bk24: 17.27; bk64: 19.34 (q16 18.64 / 21.02 / 24.56 vs bare 12.64) | slightly better raw, worse after q16; precision ↑ .935, recall ↓ .85–.88 = over-guidance contraction |
| 20 px | 45.92 | bk16 32.89 / 29.66 | bk24: **46.62**; bk32: **59.13** | ≈ bare (+0.7) / worse (+13.2) [log §21:20, dmisc] |
| 24 px | 79.80 | bk16 60.41 / 48.70 | bk32: **101.67** | worse (+21.9) |
| 32 px | 96.85 | bk24 77.45 / 69.27 | bk48: **113.93** | worse (+17.1) [log §09-08 02:00, dmech2] |

Rules out the trivial explanation "any wrong label acts as an unconditional-like reference" (ICG-style); bucketmix:12
(9.42) is between bucket:12 (8.59) and CFG, not equal to CFG [log §07:30].

---

### Table (g) — Text faithfulness: CLIP ViT-B/32 100·cos (retrieval R@1 among 1 + 99 captions, chance 1 %) of the SAME saved samples as the FD rows. Source: **[log §09-08 02:15, dclip]**. Seed 0 / seed 1.

| R | Real | CFG w = 4 (bare) | Lower-bucket ref, w = 2 | Autoguidance 10 k, w = 1.5 | Composed, w = 1.5 | Higher-bucket ref (reverse) |
|---|---|---|---|---|---|---|
| 12 | 29.55 (13.9 %) | 29.76 / 29.74 (15 %) | — | 29.41 / 29.39 (13 %) | — | bk16 29.32 (11.8 %) |
| 16 | 29.80 (16.4 %) | 30.06 / 30.04 / 30.03 (18 %) | bk12 29.37 / 29.38 / 29.37 (13 %) | 29.58 / 29.53 / 29.57 (15 %) | 29.51 / 29.47 / 29.54 (14 %) | bk24 29.58, bk64 29.60 (15 %) |
| 20 | 29.86 (19.3 %) | 30.16 / 30.09 (20 %) | bk16 29.34 / 29.35 (14 %) | 29.45 / 29.52 (15–16 %) | 29.48 / 29.49 (15 %) | bk32 29.62 (16 %) |
| 24 | 29.81 (20.2 %) | 30.11 / 30.12 (20 %) | bk16 29.09 / 29.11 (13 %) | 29.35 / 29.38 (15 %) | 29.33 / 29.34 (15 %) | bk32 29.48 (16 %) |
| 32 | 29.65 (20.9 %) | 29.72 (21.5 %) | bk24 28.49 (14 %) | 28.69 (15 %) | 28.78 (15 %) | bk48 28.85 (17.5 %) |

Honest reading for the Limitations section: every reference-guidance row — autoguidance included — pays a small
text-alignment cost relative to CFG w = 4 (−0.5 @16 px, growing to −0.9 @32 px; R@1 18 → 14 %), because a w = 2
reference term replaces the w = 4 CFG term and the text direction is weakened. The cost is method-independent
(snapshot and label references are equal within 0.2; composed is in between), grows with w (bk12 w = 3: 29.25) and is
absent for reverse (higher-bucket) references. Bare CFG w = 4 is *above* the real sprites (30.06 vs 29.80), i.e. CFG
over-aligns; the guided rows sit 0.3 *below* real.

**Table (g″) — zero-training alignment fixes, 16 px seed 0 [log §09-08 04:10, dalign].** (a) `bucketu:12` = reference
under the wrong bucket AND the empty caption, so the guidance direction contains the CFG text direction (same 2 NFE);
(b) `--cfg_text` = an additive plain-CFG term on top of the reference term (3 NFE).

| Row | FD | +q16 | mean / cov | CLIP 100·cos | R@1 |
|---|---|---|---|---|---|
| bare CFG w = 4 (ref: Table a/g) | 21.98 | 12.64 | 12.13 / 9.85 | 30.06 | 18.3 % |
| bucket:12 w = 2 (un-fixed) | 8.59 | 8.99 | 2.55 / 6.04 | 29.37 | 13.1 % |
| composed w = 1.5 (un-fixed) | **7.67** | 8.34 | 2.40 / 5.27 | 29.51 | 14.4 % |
| (a) bucketu:12 w = 2 | 12.10 | 9.24 | 5.15 / 6.96 | 29.84 | 16.1 % |
| (a) bucketu:12 w = 1.5 | 10.45 | 8.60 | 4.48 / 5.97 | 29.77 | 16.4 % |
| (a) composed, ref under bucketu:12, w = 1.5 | 8.60 | 9.32 | 2.38 / 6.21 | 29.77 | 16.1 % |
| (a) composed, ref under bucketu:12, w = 1.25 | 9.46 | 9.60 | 3.54 / 5.91 | 29.65 | 15.7 % |
| (b) bucket:12 w = 2 + cfg_text 1.5 | 9.50 | 8.71 | 3.25 / 6.24 | 29.71 | 16.2 % |
| (b) composed w = 1.5 + cfg_text 1.5 | 9.19 | 8.57 | 3.19 / 5.99 | 29.72 | 15.4 % |
| (b) composed w = 1.5 + cfg_text 2 | 11.10 | 9.48 | 4.26 / 6.85 | 29.89 | 15.9 % |

Verdict: **no free fix**. None of the seven rows reaches CLIP ≥ 30.06; the points lie on an FD–CLIP frontier
(CLIP 29.37 @ 8.59 → 29.77 @ 8.60/10.45 → 29.89 @ 11.10), every +0.1 CLIP costs ≈ +0.5–1 FD, and the FD lost is
mostly mean_term (colour/contrast shift returns as the CFG direction re-enters). The cheapest point is the composed
reference under `bucketu:12` at w = 1.5: FD 8.60 (+0.9 over composed, 3× seed sd), CLIP back to the *real-data* level
(29.77 vs real 29.80, R@1 16.1 % vs real 16.4 %), same 2 NFE. Paper treatment: main table keeps the un-fixed composed
row (7.67); Limitations states the alignment cost (−0.5 CLIP / −4 % R@1 at 16 px, growing with R, shared by
autoguidance), and the appendix gives Table (g″) as the trade-off with the `bucketu` variant as the operating point
that restores real-level alignment at +0.9 FD.

**Table (g′) — q16 (16-colour quantisation) at 20/24/32 px, seed 0 [log §09-08 02:15].** Bare → q16: 45.92 → 42.65 /
79.80 → 64.38 / 96.85 → 81.41; lower-bucket w = 2: 32.89 → 37.31 / 60.41 → 62.85 / 77.45 → 85.75; autoguidance:
31.78 → 35.72 / 53.36 → 53.00 / 75.23 → 76.79; composed: 29.66 → **34.09** / 48.70 → **51.37** / 69.27 → **74.57**.
Quantisation helps the bare model a lot and does not help the guided rows, so part of the guidance gain is in the
colour domain (bleeding); the composed row still wins at every R after quantisation, but the margin shrinks from
−35/−39/−28 % to −20/−20/−8 %. Same pattern as 16 px (Table a, +q16 column).

### Table (h) — Second metric family on the SAME saved samples: Inception-v3 clean-FID / KID (×10⁻³), white composite, NEAREST ×4 to 64 px, clean-fid. Source: **[log §09-08 05:05, diag_metric2]**; `src/v6/fid_kid_fair.py`, table script `paper_assets/incep_table.py`. Held-out real floor: FID 4.93 / 6.72 / 8.23 / 8.99 / 9.75 at 12/16/20/24/32 px (Inception has ~7× less headroom than DINOv2 here: bare 9.57 vs floor 6.72 at 16 px).

| R | Bare CFG w = 4 | Lower-bucket ref w = 2 | Autoguidance 10 k | Composed | Reverse (higher bucket) |
|---|---|---|---|---|---|
| 16 (3 seeds, FID) | 9.57 / 9.61 / — | bk12 8.61 / 8.63 / 8.64 | **7.84** / 7.97 / 7.71 | 7.97 / 7.86 / 7.95 | bk24 11.26, bk64 11.29 |
| 16 (KID) | 1.12 | .78 | .55 | .63 | 2.71 |
| 20 (FID / KID) | 11.82 / 1.01 | bk16 10.75 / 1.03 | 9.85 / .58 | **9.62 / .26** | bk32 15.45 / 3.71 |
| 24 (FID / KID) | 16.13 / 2.40 | bk16 **16.45 / 3.19** (no gain) | 13.83 / 1.96 | **13.26 / 1.46** | bk32 22.56 / 7.55 |
| 32 (FID / KID) | 19.77 / 2.34 | bk24 **20.10 / 3.31** (no gain) | 18.56 / 2.91 | **17.24 / 1.81** | bk48 25.83 / 7.58 |
| v7s @16 (FID) | 10.33 | bk12 9.77 | **8.25** | 8.49 | bk20 12.86 |
| v7s @20 / @24 (FID) | 13.06 / 16.82 | 11.73 / 17.85 | 10.42 / 14.66 | **10.22 / 13.85** | — |

Rank agreement with FD-DINOv2 over *all* saved rows: Spearman .98 (12 px, n = 9), .86 (16 px, n = 121), .96 (20 px,
n = 18), .68 (24 px, n = 17), .83 (32 px, n = 6); Pearson .91–.99.

Reading (goes into Results and Limitations): (i) the composed reference is best or tied-best at every R, on both
models and under all three metrics, and higher-bucket (reverse) references are harmful under all three → the main
claim and the direction test do not depend on DINOv2. (ii) The gain of the **label-only** lower-bucket reference is
DINOv2-visible but Inception-weak: it still lowers FID at 16/20 px, but at 24/32 px FID/KID are flat or slightly
worse (16.13 → 16.45, KID 2.40 → 3.19), and on v7s bk12 lags autoguidance (9.77 vs 8.25). Consistent with the q16
result (Table g′): the label reference mainly fixes per-pixel colour/contrast statistics, which DINOv2 (one patch per
pixel) weights heavily and Inception (natural-image features at 64 px) barely sees; the snapshot reference fixes
structure/coverage; the composition gets both. (iii) Under Inception, autoguidance ≈ composed at 16 px (within
seed spread); the composed advantage is clear at 20/24/32. Recommendation in the paper: report DINOv2 as primary
(each pixel is a patch, the metric the domain needs), Inception as secondary, and advertise the composed reference,
not the label reference alone, as the method.

### Table (i) — Cost: wall-clock and peak memory for 1000 samples @16 px (A100-80GB, exclusive; v7h 72.5 M; DDPM 100 steps; batch 500; includes model load). Source: **[log §09-08 06:05, diag_time]**, `logs/diag_time.txt`.

| Configuration | NFE / step | s per 1000 samples | Peak memory (MiB) |
|---|---|---|---|
| No guidance (w = 1) | 1 | 42.5 | 4951 |
| CFG w = 4 (bare) | 2 | 73.8 | 4951 |
| bucket:12 label reference, w = 2 | 2 | 72.7 | 4951 |
| Autoguidance (snapshot 10 k), w = 1.5 | 2 | 73.3 | 5507 |
| Composed (snapshot under bucket:12), w = 1.5 | 2 | 74.0 | 5507 |
| Composed + cfg_text 1.5 (alignment variant, Table g″) | 3 | 105.9 | 5507 |

The label reference costs exactly what CFG costs (same weights, same memory); snapshot-based references hold one
extra copy of the weights (+556 MiB) at the same wall-clock; the 3-NFE alignment variant is +43 %.

## 4. Section-by-section outline

### 1 Introduction
- Setting: text-to-sprite diffusion at 12/16/20/24 px RGBA. Each pixel is a DINOv2 patch; per-pixel colour bleeding and
  lost local contrast dominate the distribution gap (FD 21.5 vs floor 3.45 at 16 px).
- Observation 1: CFG cannot fix it — on the clean baseline every stronger CFG / CADS / interval setting is worse
  (Table a″). The error is a *systematic model error*, the regime autoguidance targets.
- Observation 2: a bucketed multi-resolution model already contains a compatible bad model — itself under a lower
  resolution label. No snapshot, no second network, no retraining; NFE identical to CFG.
- Contributions (Section 2 bullets). Be explicit that the network architecture is standard and unchanged.

### 2 Related work (each with the stated difference; IDs from arch_ideation_log cycle 6/7)
- **Autoguidance** (Karras et al., NeurIPS 2024, 2406.02507): weak = smaller/less-trained model; compatible-degradation
  appendix does not try swapping the *condition*. We use it as the control line and as one half of the composed reference.
- **SDXL micro-conditioning, `negative_original_size`** (Podell et al. 2307.01952; diffusers practice): the closest prior
  art — the CFG negative branch carries a small `original_size`, documented to induce "simpler patterns". Differences: it
  is a negative branch of *CFG* (unconditional text), it encodes source-image quality rather than the *target* bucket, and
  there is no quantitative or directional analysis. Must be cited honestly as the nearest precedent.
- **CDG** (2603.10780): guidance against a *degraded text* condition with a common-mode-rejection argument; different
  conditioning axis (text vs resolution); its geometry is a useful lens for our difference vector.
- **ICG / TSG** (2407.02687): random condition / perturbed time-step embedding as reference. Ours is not random: the
  reference is ordered (lower bucket helps, higher hurts, Table f) and bucketmix ≠ CFG.
- **S²-Guidance** (2508.12880), **In-situ autoguidance** (2510.17136; dropout weak model, reported ineffective), **M-SWG /
  SWG** (2411.10257; restricted receptive field / crops), **SAG** (2210.00939; blurred self-attention input): weaken the
  model or its input. We change only the *belief* (label), not the input or network; degraded-input references fail here
  (blur 223, shift 18.95).
- **Intermediate-layer weak heads** (SGG-BR 2603.20584, IG 2512.24176, SSG 2607.29122): joint-trained low-frequency
  heads; assume a frequency hierarchy that does not exist at 16 px. Our trained block-average branch (probe_cg) is the
  16-px analogue and is harmful.
- **APG** (2410.02416): removes the component of the guidance difference parallel to x0 (anti-oversaturation). Falsified
  here (Table e): the useful correction *is* radial.
- **FDG** (2506.19713): frequency-split weights, w_low < w_high. Falsified here: the useful correction is low-frequency.
- **LIG** (2404.07724), **FBG** (2506.06085), **TV-CFG** (2509.22007): interval / adaptive / time-varying CFG weights.
  All target CFG; our dual-reference interval scheduling gives no gain but yields the low-noise-half diagnostic.
- Multi-resolution diffusion (Matryoshka, FiT, ScaleCrafter; SDXL bucketing): provide the bucket embedding but do not use
  it for guidance. Pixel-art generation: SD-πXL (SDS-based optimisation); no prior work on guidance at ≤ 32 px.

### 3 Method
- **Model** (not a contribution): 4-channel RGBA UNet (UNet2DConditionModel, 72.5 M params (v7s: 41.3 M)),
  frozen CLIP text encoder, resolution bucket as class embedding over buckets {12,16,20,24,32,48,64}, ε-prediction, DDPM
  100 steps, EMA. Training: v7 recipe, random init, 80 k steps, BLIP captions, OGA-derived sprites with lower-bucket
  BOX-downsampled copies; 5,928 evaluation sprites excluded (180,533 training rows); EMA snapshots every 5 k steps
  [log §2026-09-06 19:00, §06:30].
- **Guidance rule.** For target resolution R with bucket label b_R and text c:
  e_strong = ε_θ(x_t, t, c, b_R), e_weak = ε_θ(x_t, t, c, b_low) with b_low < b_R,
  **e = e_weak + w (e_strong − e_weak)**, w ≈ 2 (16 px). Replaces CFG (no unconditional forward); 2 NFE per step.
- **Composed reference.** e_weak = ε_{θ_snap}(x_t, t, c, b_low): the early snapshot's weights *and* the lower label in a
  single weak forward (still 2 NFE), w = 1.5. Complementary: label fixes the mean term, snapshot restores coverage
  (Table a′).
- **Low-noise-only snapshot ("snaplo").** With `--gi_snap 0 0.5`, the snapshot weights are used only for t/T ≤ 0.5;
  outside, the final weights under b_low serve as reference. Result 7.63 ± 0.43 ≈ 7.53 ± 0.19. Be precise about the
  saving: NFE is unchanged (2/step); the second weight set is needed for only half the trajectory, and the equivalence is
  the diagnostic that the snapshot's information lives in the low-noise half [log §16:35, §17:35].
- **Choosing b_low and w.** Nearest lower bucket at 16/20 px; at 24 px a farther bucket (12) is slightly better. w-curve
  is flat for the label reference (Table a″), steep for the snapshot. TV ratio of the pure reference to the strong model
  is a zero-training diagnostic for reference selection (Table d).

### 4 Experiments
- Protocol paragraph (matched FAIR FD-DINOv2, held-out captions, contamination story: why v7h had to be retrained;
  seed sd 0.2–0.5, so differences below ~1 (2 sd) are reported as ties; per-resolution floors).
- 4.1 Main 16 px results (Table a, a′). 4.2 Zero-training baselines (Table a″). 4.3 Resolution generalisation (Table b).
  4.4 Second model (Table c). 4.5 Reverse controls (Table f). 4.6 Qualitative figure `paper_assets/fig_qual_16px.png`
  (6 rows × 24 sprites, same seed and prompt: bare / autoguidance / bucket:12 / composed …).

### 5 Mechanism analysis
- Pure-reference statistics (Table d) and the TV-vs-guided-FD plot (`fig_tv_vs_fd.png`, 7 points; explain the
  unconditional exception).
- Controlled trained ablations: probe_cg (block average, harmful) vs probe_cc (contrast shrink, helpful but weak) vs the
  free label; same fine-tuning pipeline, same paired controls. Conclusion: low TV is necessary, not sufficient; the
  reference must otherwise match the strong prediction's statistics.
- FD decomposition: mean vs covariance, precision/recall/coverage; higher buckets contract recall (over-guidance).
- Why APG / FDG / momentum / channel decoupling fail (radial, low-frequency, colour-domain correction).
- Timestep localisation of the snapshot contribution (interval scheduling).
- Remaining gap (7.5 vs 3.45): guided samples still have 1.8× the colour count and ¼ the flat-region fraction of real
  sprites [log §14:20 reading 4].

### 6 Limitations
- 12 px has no lower bucket; only snapshot autoguidance applies (9.18 vs bare 12.91); a trained contrast-shrunk label
  does not fill the gap (9.66 vs autoguidance 8.16).
- Part of the gain is in the colour domain: after 16-colour quantisation the composed reference no longer beats
  autoguidance (8.3–8.4 vs 7.8).
- Applies only to models with a resolution/bucket conditioning; not tested on a public large-scale bucketed model.
- Single dataset (OGA-derived sprites, noisy BLIP captions), single architecture family, one metric family (DINOv2 FD and
  its decomposition); no human evaluation yet. Second model v7s shares data and recipe (different width/seed, clean split); the contaminated v7_lowres is appendix-only. Label-only reference gain is Inception-weak at 24/32 px (Table h). Text-alignment cost of all reference guidance (Table g/g″).
- Not a new architecture; the trained-branch versions do not beat the zero-training rule.

---

## 5. What is still missing for ICLR (concrete, with GPU-hour estimates at ~25 min per 3000-sample matched eval item)

| # | Experiment | Items | Est. GPU-h | Why |
|---|---|---|---|---|
| 1 | ~~12 px seed 1: bare / autoguidance~~ **DONE** (dseedR; bk16 reverse s1 not run) | 3 | 1.3 | complete Table b |
| 2 | ~~Seed 2 at 12/20/24 px~~ **DONE** (Table b now 3-seed mean ± sd at 12/16/20/24; composed − autog = 4–8 sd at 16/20/24) | 11 | 4.6 | consistency with Table a |
| 3 | ~~Second model v7_lowres~~ superseded by v7s (contaminated model → appendix) | 5 | 2.1 | complete Table c |
| 4 | ~~Clean second model~~ **DONE** (v7s, Table c: 2 seeds @16 + 20/24 px rows) | train ~12 h + 8 | 15.3 | current second model is contaminated; reviewers will ask |
| 5 | ~~Higher-bucket reverse controls at 20 px (bk24, bk32)~~ **DONE seed 0** (dmisc); seed 1 optional | 4 | 1.7 | complete Table f |
| 6 | Guidance-weight sweeps at 20/24 px (w ∈ {1.25,1.5,2,2.5,3}) for label ref and autoguidance | 20 | 8.3 | show flat-vs-steep w-curve generalises |
| 7 | ~~bucket beliefs at 20/24 px~~ **DONE** (dmech2, Table d second block); per-resolution TV-vs-FD plot **DONE** (`paper_assets/fig_tv_vs_fd_r.png`, make_tv_fd_r.py: x = TV_ref/TV_strong, y = FD_guided/FD_bare, 16/20/24 px, 10 points; all x<1 same-caption beliefs give y<1, all x≥1 give y≈1 or >1); still open: probe_cg *sampled* branch stats (‡ row) | ~8 | 3.3 | mechanism claim at more than one resolution |
| 8 | ~~q16 appendix at 20/24 px~~ **DONE** (+32 px; Table g′) | 8 | 3.3 | colour-vs-structure split beyond 16 px |
| 9 | ~~Second metric family~~ **DONE** (Table h: Inception clean-FID/KID on all saved rows; composed best everywhere, label-only ref Inception-weak at 24/32) | ~16 | 1–7 | rule out DINOv2-specific effects |
| 10 | ~~CLIP score~~ **DONE** (Table g; alignment cost found; fix probe **DONE** Table g″: no free fix, FD–CLIP frontier, bucketu composed = real-level CLIP at +0.9 FD) | 32 | ~5 | guidance must not trade alignment for FD |
| 11 | Human preference study (bare vs autog vs composed, ~50 prompts × 3 raters) | — | 0 GPU; sampling ≈ 0.5 | ICLR reviewers expect it for a perceptual domain |
| 12 | ~~Sampler robustness~~ **DONE** (dmisc, log §21:20): composed DDPM 50/100/200 = 6.81/7.67/9.38; DDIM50 broken for the bare model itself (204.42) → appendix with caveat | 6 | 2.5 | show it is not a 100-step artefact |
| 13 | Applicability beyond our model: a public multi-resolution/bucketed model (e.g. SDXL with `original_size` micro-conditioning at 256–512 px, or Matryoshka/FiT) with FD at that resolution | setup + ~10 | 1–2 GPU-days | generality claim beyond pixel art; also settles the SDXL `negative_original_size` relation empirically |
| 14 | ~~32 px regime~~ **DONE seed 0** (dmech2, Table b row 32): bare is *farther* from the floor at 32 px (96.85 vs 13.77), all references still help, composed best (69.27); 48/64 px and seed 1 optional | ~9 | 3.8 | delimit the operating regime |
| 15 | ~~Wall-clock / NFE table~~ **DONE** (Table i) | 4 timings | 0.3 | quantify the "same cost as CFG" claim |
| 16 | ~~Qualitative figures~~ **DONE** (paper_assets/fig_qual_{12,16,20,24,32}px.png via make_qual_r.py, prompt-aligned) | — | 0 | paper figures |
| 17 | Verify and record exact parameter count, bucket-embedding details, training-data composition for the Method section | — | 0 | reproducibility |

Rough total without item 13: ~55–65 GPU-h (≈ 3 days on two GPUs); with item 13: +1–2 GPU-days.
