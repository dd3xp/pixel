# 5 Mechanism analysis

Why does a lower bucket label give a useful weak reference while a higher one does not, and what does the correction change? We use statistics of the *pure* weak-reference beliefs, a decomposition of FD, two trained reference branches, and two further metric families. Numbers are from `paper_outline.md` Tables a′, c, d, e, f, g′, h and log entries dmech, dmech2, dcc12, dsched, diag_metric2.

## 5.1 What the lower-resolution belief looks like

We sample each candidate reference alone (`--cfg 0`, following $e_{\text{ref}}$ only; 3000 samples, matched protocol) and measure over opaque pixels the median number of unique colours per sprite (ncol), the fraction of exactly equal adjacent pixel pairs (flat) and the mean adjacent $|\Delta\mathrm{RGB}|$ (TV). Table d, 16 px (all statistics in this section are seed 0 unless stated):

| Reference (sampled alone) | FD alone | ncol | flat | TV | FD when used as reference |
|---|---|---|---|---|---|
| Real 16 px sprites | 3.45 | 34 | .202 | 30.2 | — |
| Strong model (v7h, CFG $w=4$) | 21.98 | 73 | .030 | 32.0 | — |
| bucket:12 belief | 36.43 | 78 | .015 | 21.6 | 8.59 |
| Snapshot 10 k | 69.16 | 95 | .000 | 27.3 | 8.98 |
| Unconditional (no text) | 32.15 | 68 | .017 | 27.3 | 21.98 (via CFG $w=4$) |
| bucket:24 belief | 34.50 | 75 | .039 | 31.9 | 17.27 |
| bucket:64 belief | 327.6 | 102 | .046 | 40.9 | 19.34 |

The bucket:12 belief is *not* a simpler image in the sense of real low-resolution sprites (real 12 px sprites: median 5 colours, flat $\approx$ .51). It has *more* colours than the strong prediction (78 vs 73) and essentially no flat regions (.015), which rules out a "simplicity prior" explanation. What distinguishes it is local contrast: TV 21.6 against 32.0 — a soft, low-contrast rendering of the same sprite. The label acts as a monotone contrast knob on the same network (TV 21.6 → 31.9 → 40.9 for bucket 12 → 24 → 64), and only the low-contrast end is a useful reference.

The pattern replicates at 20 and 24 px (Table d, second block), now with the opaque-pixel fraction:

| $R$ | Reference | FD alone | opaque | ncol | flat | TV | FD as reference ($w=2$) |
|---|---|---|---|---|---|---|---|
| 20 | Real | 12.14 | .345 | 33 | .277 | 29.9 | — |
| 20 | Strong | 45.92 | .362 | 96 | .056 | 31.0 | — |
| 20 | bucket:16 | 94.02 | .395 | 107 | .030 | 21.8 | 32.89 |
| 20 | bucket:12 | 179.63 | .455 | 121 | .018 | 16.9 | 35.85 |
| 20 | bucket:24 | 66.07 | .355 | 97 | .038 | 28.5 | 46.62 |
| 24 | Real | 12.96 | .333 | 29 | .345 | 29.0 | — |
| 24 | Strong | 79.80 | .363 | 130 | .067 | 30.3 | — |
| 24 | bucket:16 | 173.39 | .425 | 142 | .028 | 18.2 | 60.41 |
| 24 | bucket:32 | 60.16 | .307 | 86 | .097 | 31.3 | 101.67 |

The references that help have TV well below the strong model (21.8, 16.9, 18.2); those that do not have TV at or above it (28.5, 31.3). The reference's *own* quality is anti-correlated with its usefulness: at 20 px bucket:24 is the best-looking belief (FD 66.07 vs 94.02 and 179.63) and the worst reference; at 24 px bucket:32 is the best-looking (60.16) and actively harmful (101.67 vs bare 79.80).

## 5.2 TV predicts the guided FD, with one instructive exception

`fig_tv_vs_fd.png` plots, for the seven 16 px references with measured statistics (the five above plus the two trained branches of §5.5), pure-reference TV against the FD obtained when that reference guides. `fig_tv_vs_fd_r.png` normalises per resolution — $x = \mathrm{TV}_{\text{ref}}/\mathrm{TV}_{\text{strong}}$, $y = \mathrm{FD}_{\text{guided}}/\mathrm{FD}_{\text{bare}}$ — for the ten bucket and snapshot references at 16/20/24 px: every same-caption belief with $x<1$ gives $y<1$, and every reference with $x \ge 1$ gives $y \approx 1$ or $y > 1$.

The single violation is the unconditional prediction: TV 27.3, as low as the snapshot, yet no gain (21.98 at CFG $w=4$, worse at higher $w$). It clarifies the rule. The unconditional branch is not a same-condition belief; its difference from the strong prediction carries the *text* direction and is applied through CFG at $w=4$ rather than as a structure-aligned reference. Low TV is a property of the pure belief, but guidance extrapolates along the *difference*, which is a contrast correction only when the two predictions agree on everything but contrast.

The same data support the structure-alignment clause. At 20 px the farthest lower bucket (12) has the lowest TV (16.9) but drifts in opacity (.455 vs .362) and is the weaker reference (35.85 vs 32.89, seed 0; 32.75 ± 0.60 over three seeds); at 32 px the farther bucket is clearly worse (bucket:16 91.26 vs bucket:24 77.45, seed 0, Table b). The one counter-example is 24 px, where bucket:12 (57.70, seed 0) marginally beats bucket:16 (60.41 seed 0; 59.32 ± 1.37 over three seeds) and the nearest bucket:20 is worst (66.20). Lower TV is better until the belief starts to drift structurally.

## 5.3 Why higher-bucket references fail

A higher bucket produces a belief with contrast at or above the strong prediction, so $e_{\text{strong}} - e_{\text{ref}}$ points *towards* lower contrast and extrapolating along it removes detail. The signature is over-guidance contraction: at 16 px the bucket:20/24/64 references raise precision to .935 and lower recall to .85–.88 (bare .907 / .902), and their samples are worse after 16-colour quantisation (q16 18.64 / 21.02 / 24.56 vs 12.64 bare) although raw FD is slightly better than bare (17.14 / 17.27 / 19.34 vs 21.98; all seed 0). At 12, 20, 24 and 32 px the higher bucket is no better than or worse than no guidance (seed-0 reverse rows 14.45; 46.62 and 59.13; 101.67; 113.93 against bare 13.02 ± 0.28; 47.04 ± 1.11; 78.96 ± 0.74; 96.85 (seed 0); Table f), and the second model repeats the pattern (bucket:20 25.31 raw vs 28.00 but 24.14 vs 13.60 after q16; precision .925 / recall .848; seed 0). The effect is ordered by the label, which rules out the reading that any wrong label acts as an unconditional-like reference (as in ICG); mixing the wrong label with the unconditional branch (bucketmix, 9.42) sits between the label reference (8.59) and CFG (21.98; seed 0).

## 5.4 What each reference corrects: FD decomposition and timestep localisation

Splitting FD into mean and covariance terms separates the two references (Table a′, 16 px seed 0; CFG 12.13 / 9.85). The label reference has the lowest mean term (2.55) but a weaker covariance term and coverage (6.04, .892); snapshot autoguidance a larger mean term (3.88) but the best covariance term and coverage (5.09, .928); the composed reference keeps the label's mean term (2.40) and most of the snapshot's covariance gain (5.27, between 5.09 and 6.04; coverage .909). The second model reproduces the split: mean term 16.48 → 5.50 (label) / 7.04 (snapshot) / 3.84 (composed), covariance 11.51 → 7.31 / 6.91 / 6.70 (Table c). The label reference removes a systematic shift of the generated distribution — the per-pixel colour and contrast error — while the snapshot restores coverage.

Interval scheduling (Table e, seed 0) localises the snapshot in time: snapshot weights only for $t/T \in [0, 0.5]$ give 7.70 (mean / cov 2.42 / 5.28), indistinguishable from the full composed rule (7.67, 2.5 / 5.2); only for $[0.5, 1]$, 8.99 (3.54 / 5.45), close to the label reference alone (8.59); $[0.2, 0.8]$, 8.84. The snapshot's information lives in the low-noise half; at high noise the 10 k snapshot and the final weights share the same layout belief. The label reference acts throughout.

The falsified components (Table e) fix the geometry of the correction: APG [arXiv 2410.02416], which removes the component of the guidance difference parallel to $x_0$, raises the composed FD from 7.67 to 9.94; frequency-decoupled guidance [arXiv 2506.19713] with $w_{\text{low}} < w_{\text{high}}$ raises the mean term monotonically (2.5 → 3.54 → 4.8 for $w_{\text{low}} = 1.5, 1.25, 1.0$); and with channel-decoupled weights FD is set almost entirely by the RGB weight (1.5 → 8.35, 2 → 11.16, 2.5 → 19.17, 3 → 32.36). The useful correction is radial, low-frequency and in the colour channels — per-pixel contrast, not high-frequency detail.

## 5.5 Controlled trained references: low TV is necessary but not sufficient

If the mechanism is "extrapolate away from a low-contrast, structure-aligned belief", a trained branch built to have those properties should work and one that has low contrast but imposes its own structure should not. We fine-tune v7h for 20 k steps with a second label set (class embedding 7 → 14), replacing the training view with probability 0.5 by a degraded view under the second label [log 2026-09-07 §13:11 (probe_cg), §19:05 (probe_cc)], and sample with $e = e_{\text{coarse}} + w\,(e_{\text{fine}} - e_{\text{coarse}})$. Each probe is compared with the label reference and snapshot autoguidance *on the same fine-tuned weights* (Table d, paired-control block), so fine-tuning is not a confound.

*Block average (probe_cg).* Degraded view = 2×2 block average, nearest-neighbour upsampled. Its training target has TV 14.2 but flat .605 and ncol 21 — a block grid the strong prediction lacks. [Note: these are statistics of the block-averaged real sprites the branch was trained on; the sampled branch was not measured — open item in the outline.] As reference it is *harmful*: 21.53 / 35.22 / 59.38 for $w = 1.5 / 2 / 3$ against 19.17 bare on the same weights, mean term 13.32 → 42.05 for $w = 1.5 \to 3$ (bare 9.95) [log §13:11], while the paired label reference (10.12) and snapshot (9.10) work as usual.

*Contrast shrink (probe_cc).* Degraded view = RGB shrunk 0.6× towards the per-image opaque mean: structure-aligned, no grid. Sampled alone: FD 41.37, ncol 64, flat .034, TV 15.9. As reference it is *effective* — 14.71 at $w = 1.5$ vs 20.63 bare on the same weights (19.98 / 44.90 at $w = 2, 3$) — so replacing the block grid by a structure-aligned degradation flips the sign of the effect, confirming the direction under a controlled change. But it is far weaker than the free label on the same weights (9.43) and the paired snapshot (8.77), with a mean term (8.1) of the order of bare CFG's on the same weights (10.97) rather than the same-weights label reference's (3.25) [log §19:05]. Its statistics show why: TV 15.9 is below the bucket:12 belief's 21.6, yet ncol falls to 64 and flat rises to .034 — an additional "fewer colours, flatter" bias that extrapolation amplifies in reverse. It also fails to fill the 12 px gap: 9.66 ($w = 1.25$) vs 8.16 for snapshot autoguidance and 11.78 bare on the same weights.

Three references with TV 14.2 / 15.9 / 21.6 thus span harmful / weakly effective / effective, in reverse order of TV. Lower TV than the strong prediction is necessary (bucket:24/64 fail) but not sufficient (both trained branches under-perform the label). The effective reference must differ from the strong prediction *only* in local contrast, with colour count, opacity and structure otherwise matched — which the label satisfies because it changes the belief and nothing else, and which no hand-designed degradation we tried does.

## 5.6 Where the label reference's gain lives: quantisation and a second metric family

Two measurements on the *same* saved samples show that the label reference mainly corrects per-pixel colour and contrast statistics, whereas the snapshot corrects structure and coverage.

*16-colour quantisation (q16; Tables a, g′; seed 0).* Quantising every sample to 16 colours before FD helps the bare model substantially (21.98 → 12.64 / 45.92 → 42.65 / 79.80 → 64.38 / 96.85 → 81.41 at 16/20/24/32 px) and does not help, or hurts, the guided rows (label reference 8.59 → 8.99 / 32.89 → 37.31 / 60.41 → 62.85 / 77.45 → 85.75; composed 7.67 → 8.34 / 29.66 → 34.09 / 48.70 → 51.37 / 69.27 → 74.57). At 16 px autoguidance and composed swap order after q16 (7.82 vs 8.34), and the composed margin over bare shrinks from −35 / −39 / −28 % to −20 / −20 / −8 % at 20/24/32 px. Part of the gain is in the colour domain — bleeding that a 16-colour palette partly removes — and that part grows with resolution.

*Inception-v3 clean-FID / KID (Table h).* On natural-image features after nearest-neighbour upsampling to 64 px, the composed reference is best or tied-best at every resolution and on both models (FID 7.97 vs autoguidance 7.84 at 16 px, within the 3-seed spread; 9.62 vs 9.85, 13.26 vs 13.83, 17.24 vs 18.56 at 20/24/32 px), and higher-bucket references are harmful here too (11.26 / 15.45 / 22.56 / 25.83 vs bare 9.57 / 11.82 / 16.13 / 19.77). The *label-only* reference, however, is Inception-weak: it lowers FID at 16 and 20 px (9.57 → 8.61, 11.82 → 10.75) but is flat or slightly worse at 24 and 32 px (16.13 → 16.45, KID 2.40 → 3.19; 19.77 → 20.10, KID 2.34 → 3.31), and on the second model it lags autoguidance (9.77 vs 8.25). Rank agreement between the metric families is high (Spearman .98 / .86 / .96 / .68 / .83 at 12/16/20/24/32 px), so this is a difference in sensitivity, not a contradiction: DINOv2 at these sizes sees one patch per pixel and weights per-pixel colour statistics heavily; Inception at 64 px barely sees them. Both readings agree with §5.4: the label reference fixes the mean shift in colour space, the snapshot fixes coverage, and the composed reference is the one to advertise.

## 5.7 What remains

After guidance the sample statistics move towards the real distribution without reaching it (Table d, last row): ncol 73 → 61 / 64 / 60.5 (label / snapshot / composed) against 34 real; flat .030 → .045 / .041 / .044 against .202. Guided samples still have about 1.8× the colour count and a quarter of the flat-region fraction of real sprites; the remaining gap (7.53 vs the 3.45 floor) has the form of residual colour mixing. For precision: the label reference alone does not lower the TV of the *guided* samples (32.4 vs 32.0 at 16 px; 32.5 vs 31.0 at 20 px), so "restoring contrast" should be read as extrapolating away from a low-contrast belief, not as a monotone change of the TV statistic.
