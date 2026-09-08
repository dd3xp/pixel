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
snapshot: 8.7); composing the two references reaches 7.5 (floor 3.45). The effect transfers to 20 and 24 px and to a
second model, is strictly directional (higher-bucket references hurt), and is explained by a controlled mechanism study:
an effective weak reference must be lower in total variation *and* structure-aligned, a condition that explicit low-pass
branches violate.

---

## 2. Contribution bullets (honest)

1. **A zero-training guidance mechanism for bucketed multi-resolution diffusion models.** The weak reference is the model's
   own prediction under a lower-resolution bucket label (same weights, same x_t, same text); guidance replaces CFG at
   identical cost (2 forwards/step). At 16 px it matches early-snapshot autoguidance (8.52 ± 0.29 vs 8.73 ± 0.26 over 3
   seeds) with no snapshot, and composing both references gives 7.53 ± 0.19, a 65 % reduction from the CFG baseline
   21.52 ± 0.47 [log §17:35]. This is not a new network architecture; the model, data and training recipe are unchanged.
2. **Generality and directionality evidence.** The gain reproduces at 20 px (45.9 → 29.7) and 24 px (79.8 → 48.7), on a
   second independently trained model (16.66 → 7.40), and is directional at three resolutions: *higher*-bucket references
   are worse than no guidance at 12 and 24 px and far weaker at 16 px [log §14:05, §15:10, §16:35].
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
| **Cross-res self-guidance w = 2 (ours)** | bucket:12, same weights | **no** | 8.59 | 8.20 | 8.76 | **8.52 ± 0.29** | 8.15 / TBD / 8.57 |
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

### Table (b) — Resolution generalisation, matched FD at the native resolution (fd_fair --size R). Sources: **[log §14:05]** (seed 0, 12/16/20 px), **[log §15:10]**, **[log §16:35]** (24 px completion), **[log §19:05 + §19:35]** (dseedR seed 1). Seed-1 values for 20/24 px as given in the task brief (dseedR).

Weights v7h throughout; snapshot = 10 k EMA; "lower" = nearest lower bucket unless stated.

| R | Floor | CFG w = 4 (bare) | Lower-bucket ref, w = 2 | Other lower bucket, w = 2 | Autoguidance 10 k, w = 1.5 | **Composed (lower + 10 k), w = 1.5** | Higher-bucket ref, w = 2 (reverse control) |
|---|---|---|---|---|---|---|---|
| 12 | 3.37 | 12.91 / 12.81 | — (no lower bucket exists) | — | **9.18 / 8.70** | — | bk16 **14.45** (worse than bare) (s0) |
| 16 | 3.45 | 21.98 / 21.05 | bk12 **8.59** / 8.20 | — | 8.98 / 8.73 | **7.67 / 7.32** | bk20 17.14, bk24 17.27, bk64 19.34 |
| 20 | 12.14 | 45.92 / 48.14 | bk16 **32.89** / 32.09 | bk12 35.85 (s0) | 31.78 / 31.24 | **29.66 / 28.72** | bk24 **46.62** (≈ bare), bk32 **59.13** (worse) (s0) |
| 24 | 12.96 | 79.80 / 78.68 | bk16 **60.41** / 57.78 | bk12 57.70, bk20 66.20 (s0) | 53.36 / 54.77 | **48.70 / 47.86** (bk16 + 10 k) | bk32 **101.67** (worse than bare) |
| 32 | 13.77 | 96.85 | bk24 **77.45** | bk16 91.26 | 75.23 | **69.27** (bk24 + 10 k) | bk48 **113.93** (worse than bare) |

Cell format: seed 0 / seed 1 (32 px seed 0 only, **[log §09-08 02:00, dmech2]**). Caption points: (i) relative gain of the
composed reference: 16 px −65 %, 20 px −35 %, 24 px −39 %, 32 px −28 % (seed 0); (ii) the bare model is much further from
the floor at 20/24/32 px (gap 33.8 / 66.8 / 83.1 vs 18.5 at 16 px) because training data is 16 px-dominated
[log §14:05 reading 3]; (iii) at 24 px a *farther* lower bucket (bk12 57.70) is slightly better than the nearest
(bk16 60.41), but at 32 px the farther bucket is clearly worse (bk16 91.26 vs bk24 77.45) — the belief must stay
structurally aligned, see Table d; (iv) 12 px has no lower bucket, so only snapshot autoguidance applies — a real
limitation, stated as such; (v) the ordering bare > lower-bucket ≈ autoguidance > composed holds at every resolution
that has a lower bucket (16/20/24/32); at 32 px autoguidance (75.23) edges the label reference (77.45).

### Table (c) — Second model (v7_lowres). Source: **[log §14:05]**; bare/q16 from **[log §2026-09-06 19:00, cycle 5 (3)]**.

v7_lowres = the original v7 model (initialised from v6e10, 60 k steps, buckets [12,16,20,24,32,48,64]). Caveat that must
be in the caption: its training set contains the evaluation sprites (memorisation contamination; 8.4 % of its samples are
near pixel-exact copies), so only the *relative* drop is meaningful.

| Model @16 px | CFG w = 4 (bare) | bucket:12 ref, w = 2 | Autoguidance | Composed | Δ (bare → bucket:12) |
|---|---|---|---|---|---|
| v7h (clean, Table a) | 21.98 (q16 12.64) | 8.59 (q16 8.99) | 8.98 | 7.67 | −13.4 (−61 %) |
| v7_lowres (contaminated) | 16.66 (q16 11.29) | **7.40** (q16 7.86) | **TBD** | **TBD** | −9.3 (−56 %) |

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
- **Model** (not a contribution): 4-channel RGBA UNet (UNet2DConditionModel, ~72 M params — TBD verify exact count),
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
  seed sd 0.2–0.5; decision rule "differences < 4 are inconclusive" from autonomy_state; per-resolution floors).
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
  its decomposition); no human evaluation yet. Second model shares data and lineage and is contaminated.
- Not a new architecture; the trained-branch versions do not beat the zero-training rule.

---

## 5. What is still missing for ICLR (concrete, with GPU-hour estimates at ~25 min per 3000-sample matched eval item)

| # | Experiment | Items | Est. GPU-h | Why |
|---|---|---|---|---|
| 1 | ~~12 px seed 1: bare / autoguidance~~ **DONE** (dseedR; bk16 reverse s1 not run) | 3 | 1.3 | complete Table b |
| 2 | Seed 2 at 12/20/24 px for bare / lower / autog / composed (3-seed mean ± sd at every resolution) | 11 | 4.6 | consistency with Table a |
| 3 | Second model v7_lowres: autoguidance and composed (seed 0 + 1), bucket:12 seed 1 | 5 | 2.1 | complete Table c |
| 4 | **Clean second model**: retrain a second architecture/size variant (e.g. different width or bucket set) with the evaluation split excluded, then 4 rows × 2 seeds | train ~12 h + 8 | 15.3 | current second model is contaminated; reviewers will ask |
| 5 | ~~Higher-bucket reverse controls at 20 px (bk24, bk32)~~ **DONE seed 0** (dmisc); seed 1 optional | 4 | 1.7 | complete Table f |
| 6 | Guidance-weight sweeps at 20/24 px (w ∈ {1.25,1.5,2,2.5,3}) for label ref and autoguidance | 20 | 8.3 | show flat-vs-steep w-curve generalises |
| 7 | ~~bucket beliefs at 20/24 px~~ **DONE** (dmech2, Table d second block); still open: probe_cg *sampled* branch stats (‡ row) and TV-vs-FD plot per resolution (no GPU) | ~8 | 3.3 | mechanism claim at more than one resolution |
| 8 | q16 appendix at 20/24 px for the four main rows | 8 | 3.3 | colour-vs-structure split beyond 16 px |
| 9 | Second metric family on saved samples: Inception FID / KID and precision–recall at all resolutions (re-scoring only if samples were kept; otherwise regenerate) | ~16 | 1–7 | rule out DINOv2-specific effects |
| 10 | Text-faithfulness: CLIP score of the four main rows at 12/16/20/24 (2 seeds) | 32 (cheap, ~10 min each) | ~5 | guidance must not trade alignment for FD |
| 11 | Human preference study (bare vs autog vs composed, ~50 prompts × 3 raters) | — | 0 GPU; sampling ≈ 0.5 | ICLR reviewers expect it for a perceptual domain |
| 12 | ~~Sampler robustness~~ **DONE** (dmisc, log §21:20): composed DDPM 50/100/200 = 6.81/7.67/9.38; DDIM50 broken for the bare model itself (204.42) → appendix with caveat | 6 | 2.5 | show it is not a 100-step artefact |
| 13 | Applicability beyond our model: a public multi-resolution/bucketed model (e.g. SDXL with `original_size` micro-conditioning at 256–512 px, or Matryoshka/FiT) with FD at that resolution | setup + ~10 | 1–2 GPU-days | generality claim beyond pixel art; also settles the SDXL `negative_original_size` relation empirically |
| 14 | ~~32 px regime~~ **DONE seed 0** (dmech2, Table b row 32): bare is *farther* from the floor at 32 px (96.85 vs 13.77), all references still help, composed best (69.27); 48/64 px and seed 1 optional | ~9 | 3.8 | delimit the operating regime |
| 15 | Wall-clock / NFE table: CFG vs bucket-ref vs composed vs snaplo (memory and time for 3000 samples) | 4 timings | 0.3 | quantify the "same cost as CFG" claim |
| 16 | Qualitative figures at 12/20/24 px (same seed/prompt grids as fig_qual_16px) | 4 small samplings | 0.5 | paper figures |
| 17 | Verify and record exact parameter count, bucket-embedding details, training-data composition for the Method section | — | 0 | reproducibility |

Rough total without item 13: ~55–65 GPU-h (≈ 3 days on two GPUs); with item 13: +1–2 GPU-days.
