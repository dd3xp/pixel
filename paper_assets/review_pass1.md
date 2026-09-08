# Review pass 1 — ICLR AC-level read of the six drafts

Scope: `draft_{intro,related,method,mechanism,experiments,limitations}.md` checked against `paper_outline.md`
(Tables a–i) as source of truth. Part A is a number-by-number cross-check; Part B is the reviewer critique and the
length plan. Nothing in the drafts or the outline was edited.

Word counts (whitespace tokens): intro 1125 · related 935 · method 1550 · experiments 3851 · mechanism 2305 ·
limitations 745 · **total 10,511** against a target of ~5,950 for the main text (700/550/1100/2000/1200/400).

---

## Part A — Numeric cross-check

Method: every numeral in each draft was located in the outline (Tables a, a′, a″, b, c, d, e, f, g, g′, g″, h, i, §3–§5
text) or recomputed when it is a ratio/difference of outline numbers. ~640 numeric tokens checked. Everything not
listed below matches the outline exactly (all of Tables 1, 3, 4, 5, 6, 7, 8, 9, 10 in `draft_experiments.md`, both
tables in `draft_mechanism.md`, and the CLIP/Inception/cost/q16 sentences reproduce the outline cells verbatim).

### A.1 Mismatches and numbers with no outline source

| # | Location | Draft says | Outline says | Severity |
|---|---|---|---|---|
| 1 | `draft_mechanism.md:47` | higher-bucket refs at 16 px "raise precision to **.935–.939** and lower recall to **.829–.882**" | Table f: "precision ↑ **.935**, recall ↓ **.85–.88**". `.939` is the *real-floor* precision in Table a′; `.829` appears nowhere. | **Mismatch** — fix or source from log |
| 2 | `draft_mechanism.md:59` | probe fine-tuning: "**20 k steps**", "class embedding **7 → 14**", "probability **0.5**" | **Not found.** Outline never states the probe training recipe. | Needs a log citation |
| 3 | `draft_mechanism.md:61` | probe_cg "mean term **13.32 → 42.05**" | **Not found** (no mean/cov terms for probe_cg in Table d or its paired-control block). | Needs a log citation |
| 4 | `draft_mechanism.md:63` | probe_cc "mean term (**8.1**) of the order of bare CFG's (**10.97**) rather than the label reference's (**3.25**)" | **Not found.** Note `3.25` *does* occur in the outline, but as the mean term of `bucket:12 w=2 + cfg_text 1.5` (Table g″), a different quantity — check this is not a copy error. | Needs a log citation; possible cross-table contamination |
| 5 | `draft_experiments.md:51` (Table 2, last row) and `:34` | 20 px `bucket:16` sweep w = 1.25/1.5/2/2.5/3 → **40.96 / 37.48 / 32.89 / 35.15 / 38.53** | **Not found.** Outline §5 item 6 lists "guidance-weight sweeps at 20/24 px" as *still open*; only the w = 2 value (32.89) exists. `draft_method.md:63` says "[TBD: weight sweeps at 20/24 px are open]". | **Cross-draft contradiction** (experiments reports what method says is missing). Either add to outline with a log source, or drop the row. |
| 6 | `draft_intro.md:14`, `draft_experiments.md:69,76` | v7s @16 "28.00 → 12.80 → 10.54, **−63 %**" | Outline Table c also says −63 %, but 1 − 10.54/28.00 = **−62.4 %**. −63 % only holds on the 2-seed mean (27.875 → 10.41 = −62.7 %). | Arithmetic: say −62 % with seed-0 numbers, or quote the 2-seed mean |
| 7 | `draft_method.md:55` | label reference costs CFG "**within 1 %** wall-clock" | 72.7 vs 73.8 s = **1.5 %** | Arithmetic; say "within 2 %" or "1.5 % faster" |
| 8 | `draft_related.md:13`, `draft_limitations.md:7` | Inception has "about **7×** less headroom" | (21.98 − 3.45)/(9.57 − 6.72) = 18.53/2.85 = **6.5×**. Outline says "~7×" too. | Arithmetic; "≈ 6.5×" or "roughly 6–7×" |
| 9 | `draft_experiments.md:55` | "composed − autoguidance gap is **4–8** seed sd at 16/20/24 px" | Recomputed from Table b: 16 px 1.20/(0.19–0.26) = 4.6–6.3 sd; 20 px 2.67/(0.29–0.69) = 3.9–9.2; 24 px 5.92/(0.43–0.72) = **8.2–13.8**. Outline item 2 says "4–8 sd" — the outline is what is off. | Understated at 24 px; say "≥ 4 sd" |
| 10 | `draft_experiments.md:30` | "the composition roughly **halves** both terms (mean 2.40, cov 5.27)" | Outline a′ reading: "takes roughly half of each" (i.e. half of each reference's *benefit*). Numerically 2.40 vs label 2.55 / snapshot 3.88 and 5.27 vs 5.09 / 6.04 — nothing is halved. | Wording error that a reviewer will read as a numeric claim |
| 11 | `draft_experiments.md:11` | CFG "w = 4 (**the best CFG weight**; §4.3)" | Table a″ contains only w = 4/7/10 and CADS/interval variants. No w < 4 and no w = 1 (unguided) FD anywhere in the outline. | **Unsupported claim** (see B.1) |
| 12 | `draft_experiments.md:57–64` (Table 3 header) | column "**Nearest** lower bucket, w = 2" contains `bk16` at 24 px | Nearest lower bucket at 24 px is 20; Table b lists bk20 = 66.20 under "other lower bucket" and uses bk16 (59.32) as the headline. | Mislabelled column; also a selection-on-test issue (see B.6) |

### A.2 Same quantity, different values across drafts (both values exist in the outline, but the reader sees two numbers for one cell)

| Quantity | Value A (where) | Value B (where) | Cause |
|---|---|---|---|
| 12 px bare CFG | 13.02 ± 0.28 (`experiments.md:61` Table 3; `limitations.md:9`) | 12.91 (`experiments.md:88` Table 5; `intro.md:15`; `mechanism.md:47`) | 3-seed mean vs seed 0 |
| 12 px autoguidance | 8.54 ± 0.73 (Table 3; limitations) | 9.18 (Table 5) | same |
| 20 px bare CFG | 47.04 ± 1.11 (Table 3; `intro.md:14`) | 45.92 (Table 2 caption, Table 5, `mechanism.md` table, `mechanism.md:71`) | same |
| 20 px bk16 label ref | 32.75 ± 0.60 (Table 3; intro) | 32.89 (Tables 2, 5; mechanism) | same |
| 24 px bare CFG | 78.96 ± 0.74 (Table 3; intro) | 79.80 (Table 5; mechanism; `intro.md:15`) | same |
| 24 px bk16 label ref | 59.32 ± 1.37 (Table 3) | 60.41 (Table 5; `experiments.md:55`; mechanism; method `:35`) | same |
| 24 px composed | 48.23 ± 0.43 (Table 3) | 48.70 (Table 5; mechanism `:71`) | same |
| 20 px composed | 28.90 ± 0.69 (Table 3) | 29.66 (Table 5; mechanism) | same |
| 16 px composed headline | 7.53 ± 0.19 (Tables 1, 3; intro) | 7.67 (Tables 4, 5, 8; all of §5; `experiments.md:108`) | same |
| Higher bucket "worse than no guidance" at … | 12, 24, 32 px (`intro.md:15`) | 12, 20 (bk32), 24, 32 px (`mechanism.md:47`, `experiments.md:82`, outline contribution 2) | intro drops 20 px |
| 20/24 px weight sweep | "open" (`method.md:63`) | reported (`experiments.md:51`) | see A.1 #5 |

Recommendation: pick one convention per table (3-seed mean where available, seed 0 otherwise) and state in every caption
which it is; never quote a seed-0 value in prose for a cell that has a 3-seed mean in a main table.

### A.3 Checked and consistent (so you can stop worrying)

All of Table a/a′/a″ values in Tables 1–2 and §3; Table b/c/f in Tables 3–5; Table d (both blocks) in §5.1; Table e
interval/APG/FDG/channel-decoupled values in §5.4 and Table 2; Table g/g″ in Table 7/8 and Limitations; Table g′ in
§5.6; Table h in Table 9 and §5.6; Table i in Table 10 and §3.5; floors 3.37/3.45/12.14/12.96/13.77; percentages
−65/−35/−39/−28 % (recomputed: 0.650/0.354/0.390/0.285 ✓), q16 margins −20/−20/−8 % (0.201/0.202/0.084 ✓), −43 % for v7s
@20 (0.427 ✓), gaps 18.5/33.8/66.8/83.1 ✓, +556 MiB ✓, +43 % ✓, 1.8× colour count (60.5/34 = 1.78 ✓), "a quarter" flat
fraction (.044/.202 = 0.22 ✓), CLIP −0.5 @16 (30.06 → 29.51 = 0.55 ✓) and −0.9 @32 (29.72 → 28.78 = 0.94 ✓), +0.9 FD
(8.60 − 7.67 = 0.93 ✓), R@1 −4 % (18.3 → 14.4 ✓).

---

## Part B — Reviewer critique (ranked by probability of driving a 5 / reject)

Format per item: (i) weakness, (ii) where it lives, (iii) cheapest concrete fix. GPU estimates assume the outline's
~25 min per 3000-sample matched evaluation on one A100 and that all headline samples are already saved.

### B.1 The CFG baseline is only swept *upwards*, and the guided rows pay an alignment cost — so the headline comparison may be a guidance-strength confound
(i) Table 2 shows CFG at w = 4/7/10 only. No unguided (w = 1) FD is reported anywhere, and no w ∈ {1.5, 2, 3}. Yet §4.1
declares w = 4 "the best CFG weight" and §4.9 shows every guided row sits 0.5 CLIP *below* CFG w = 4 while CFG w = 4 sits
*above* real data (30.06 vs 29.80, "over-aligns"). A reviewer will immediately ask: what is FD for CFG at the weight
whose CLIP matches the guided rows (≈ 29.5)? If CFG w = 2 gives, say, FD 12, the −65 % becomes −35 % at matched
alignment and the paper's central number is inflated. Right now the paper cannot answer this.
(ii) `draft_experiments.md` §4.1 (line 11), §4.3 (Table 2), §4.9; `draft_intro.md` line 5 ("CFG does not repair this").
(iii) **Run CFG w ∈ {1, 1.5, 2, 3} at 16 px, seed 0, plus CLIP on them** (4 items ≈ 1.7 GPU-h; CLIP is minutes). Add
the points to the FD–CLIP frontier of Table 8 so that CFG and the reference methods are compared as Pareto curves, and
report the guided-vs-CFG gap *at matched CLIP*. If CFG w < 4 is worse on FD (likely, since the error is
mean-regression), this becomes one of the strongest figures in the paper; if not, you need to know now.

### B.2 "Mechanism" is correlational and thin; the paper says "mechanism" 30+ times
(i) The evidence is 7 points at 16 px (one of which, probe_cg, has *no measured* sampled statistics — its row carries
the stats of its training target) and 10 normalised points across 16/20/24 px, all hand-chosen references, with one
exception (unconditional) explained away after the fact by adding a clause ("must be same-condition") that was not
part of the original rule. "Necessary" is inferred from two failures per resolution. There is no intervention: nothing
manipulates TV while holding structure fixed *on the same network at sampling time*. A diffusion-savvy AC will call
this a characterisation, not a mechanism, and will note that the strongest word in the title bullets ("explained by a
controlled mechanism study") is not earned.
(ii) `draft_mechanism.md` §5.1–5.2, §5.5; `draft_intro.md` bullet 4; `draft_method.md` §3.6; outline abstract.
(iii) Three cheap things. (a) **Zero-training interventional reference**: at sampling time, convert e_strong to x̂₀,
shrink RGB by factor s ∈ {0.6, 0.8} towards the per-image opaque mean, convert back to ε, and use *that* as e_ref
(exactly probe_cc's degradation but applied to the strong prediction itself, so structure alignment is guaranteed by
construction and TV is the only knob). 2 factors × 2 weights = 4 items ≈ 1.7 GPU-h. If it matches bucket:12 the
mechanism claim is real and stronger than anything in the draft; if it fails, the honest story is "the label does
something beyond contrast" and the paper should say so. (b) Measure probe_cg's *sampled* branch stats (`--cfg 0`,
1 item, 25 min) so the 7-point plot has 7 measured points. (c) Rename: "necessary conditions for an effective
reference" / "characterisation", and remove "explained by" from the abstract.

### B.3 Novelty relative to autoguidance, ICG/CDG and the SDXL `negative_original_size` practice
(i) The rule is autoguidance with the condition swapped instead of the weights, which Karras et al. explicitly frame
as "any compatible degradation"; ICG/CDG already swap conditions (random label / degraded text); and SDXL users have
passed a small `original_size` to the negative branch since 2023 with the same stated effect ("simpler patterns"). The
draft handles this honestly in §2/§3.4 but only by *assertion* ("no quantitative or directional analysis exists").
Expect: "incremental: a known trick, measured on a private model". PAG (Ahn et al. 2024, perturbed self-attention) and
SEG (Hong 2024) — the most-used zero-training self-reference guidance methods — are not cited or compared at all;
their absence will be noticed by any guidance reviewer.
(ii) `draft_related.md` §2 paragraphs 2–3; `draft_method.md` §3.4; `draft_intro.md` bullet 1.
(iii) Zero GPU: you *already ran* the SDXL-style analogue — `bucketu:12` (lower bucket + empty caption, i.e. the
label in the CFG negative branch) gives 10.45 (w = 1.5) / 12.10 (w = 2) vs 8.59 for the same-caption label reference
(Table g″). Promote this from "alignment fix" to "the negative_original_size analogue is 1.9–3.5 FD worse than keeping
the caption fixed"; that is the quantitative and directional analysis you say does not exist. Then **add a PAG
baseline** (identity self-attention map in the weak forward, w ∈ {1.5, 2}, 2 items ≈ 1 GPU-h) to Table 2 — it is the
one zero-training competitor an AC will demand, and it also costs 2 NFE.

### B.4 The novel component is the metric-fragile one; the metric-robust result needs the snapshot
(i) The selling point is "no second model to store". But the label-only reference is the row that (a) is Inception-flat
or worse at 24/32 px, (b) lags autoguidance on v7s under Inception (9.77 vs 8.25), (c) loses most of its advantage
under 16-colour quantisation, and (d) is beaten by autoguidance by 4–8 sd at 24 px (59.32 vs 54.15) and by 1.2 at
20 px, so "label ≈ autoguidance at every resolution" (intro bullet 2, §4.4) is true only at 16 px. The row that is
robust under all three metrics (composed) stores a snapshot, which is exactly what the abstract says you avoid. A
reviewer will phrase this as "the zero-storage claim and the metric-robust claim are about different methods".
(ii) `draft_intro.md` bullets 1–2 and abstract; `draft_experiments.md` §4.4 line 55, §4.10; `draft_mechanism.md` §5.6;
`draft_limitations.md` ¶2.
(iii) Re-frame, not re-run: make the composed reference the method in the abstract and Table 1, and present the
label-only reference as the *ingredient* ("the label is a free reference that closes 2/3 of the gap and composes with
a snapshot"). Replace "≈" with "matches at 16 px and is within 1.2–5 FD elsewhere". Add a **third feature extractor on
the saved samples** (CLIP ViT-B/32 image features FD, or DINOv2-B; feature extraction only, < 1 GPU-h for every saved
row) so "does not depend on the extractor" rests on three families instead of two that disagree on the novel row.

### B.5 Generality: two sibling in-house models, one dataset, and the 20–32 px regime is undertrained
(i) v7h and v7s share data, captions, recipe, architecture family, bucket set and evaluation split; they differ in width
and seed. The 20/24/32 px "generalisation" happens where the bare model is 3–7× above the floor (47/79/97 vs 12–14)
because the data are 16 px-dominated — i.e. the method is shown to help most where the model is worst, which is the
autoguidance regime by construction, not evidence about a well-trained model. No public bucketed model is tested (SDXL
micro-conditioning, Matryoshka, FiT). Reviewers will score the "generality" bullet as unsupported.
(ii) `draft_intro.md` bullet 2; `draft_experiments.md` §4.4–4.5; `draft_limitations.md` ¶5.
(iii) Two cheap moves. (a) Reword: "reproduces on a second model of the same family" and "in the low-data resolutions the
gap is larger and the gain is larger"; drop the word "generality" from the bullet title. (b) **Use the saved EMA
snapshots as *strong* models**: run bare CFG vs bucket:12 reference at 16 px with the 20 k / 40 k / 60 k snapshots as θ
(3 × 2 items ≈ 2.5 GPU-h). If the effect exists throughout training, it is a property of the bucketed parameterisation
rather than of one converged checkpoint — a much better generality argument than v7s, and free of new training. Keep the
SDXL test in future work, explicitly.

### B.6 Hyper-parameters (w, b_low, snapshot step) are selected on the same 3000 captions used for reporting; Table 3 mislabels b_low
(i) w ∈ {1.5, 2, 2.5, 3} and the snapshot step (5/10/20/40 k) are swept on the reporting set; at 24 px the reported
"lower bucket" is bk16 (59.32) while the *nearest* lower bucket bk20 gives 66.20 and bk12 gives 57.70 — the choice
was made after looking at the test FD, and Table 3's column header says "Nearest lower bucket". There is no validation
split. An AC will note that the same flat w-curve you praise (10.21/8.59/11.20/13.87) also means the reported value
is the minimum over four looks at the test set.
(ii) `draft_experiments.md` §4.1, Table 3 header (line 59), §4.3; `draft_method.md` §3.2 "Choice of b_low".
(iii) **Split the 3000 held-out captions 1500/1500 (selection/report) and recompute FD from the saved samples** —
feature extraction only, no new sampling, well under 1 GPU-h for all rows. Report selection-half w/b_low and the
report-half FD. Fix the Table 3 header to "Lower bucket used (nearest unless stated)" and say in §3.2 that b_low = 16 at
24 px was chosen on FD; better, adopt one rule (nearest) everywhere and put bk12@24 in the appendix.

### B.7 Statistics: seed 0 only for most of the paper, no estimator CI, and two numbers per cell
(i) 3 seeds exist only for 16/20/24 px headline rows. Seed 0 only: 32 px, v7s @20/24, every reverse control, every
mechanism table, all of Table e, the probes, q16 at 20–32, sampler robustness. Several claims rest on differences of
1–3 FD at seed-0 (bk12 57.70 vs bk16 60.41 "slightly better"; autog 75.23 vs label 77.45 "edges"; bk24@20 46.62 vs
45.92 "≈ bare") when the 20/24 px seed sd is 0.6–1.4. The paper reports seed-to-seed sd but never the FD estimator's
own uncertainty at n = 3000 (bootstrap), so the two are conflated. And the same cell appears as 12.91 and 13.02, 45.92
and 47.04, 7.53 and 7.67 in adjacent tables (A.2).
(ii) All of `draft_experiments.md`; `draft_mechanism.md` §5.1–5.3.
(iii) **Bootstrap CIs from saved features** (CPU only, hours at most) for every table cell; seed 1 for the 32 px row
(bare/label/autog/composed = 4 items ≈ 1.7 GPU-h); one convention per table (A.2). Mark seed-0 differences below 2 sd
as ties in the text — the outline already promises this (§4 protocol paragraph) but the drafts do not honour it.

### B.8 Metric validity: "one DINOv2 patch per pixel" is only true at 16 px, and the metric may reward exactly what the method changes
(i) 224/14 = 16, so the patch-per-pixel argument (intro ¶1, related ¶6) holds at 16 px only; at 12/20/24/32 px a
nearest-upsampled 224 input has patches straddling 1.3–2.3 pixels, and the drafts never say how R ≠ 16 is fed to DINOv2.
Separately, the label reference's gain is concentrated in per-pixel colour statistics that DINOv2 "weights heavily"
(your words), collapses under q16 and is invisible to Inception at 24/32 — a sceptic reads this as "the metric is
tuned to the failure mode the method fixes". Without a human study (none yet; outline item 11) this is hard to rebut.
(ii) `draft_intro.md` ¶1; `draft_related.md` ¶6; `draft_experiments.md` §4.1, §4.10; `draft_limitations.md` ¶2–3.
(iii) One sentence in §4.1 stating the exact resize/padding used per R. Then the **human preference study** (outline
item 11: bare vs autog vs composed, ~50 prompts × 3 raters, 0 GPU beyond samples already saved) — for a perceptual
domain at ICLR this is the single most rating-relevant missing item, and it is the one that settles the FD–CLIP
trade-off in B.1 too. Also make the metric-robust ordering (composed best under DINOv2, Inception FID, KID) a stated
main claim and demote the label-only numbers.

### B.9 Sampler fragility: 50 steps beats the headline and DDIM is broken
(i) §4.8 reports composed at 50/100/200 DDPM steps = 6.81/7.67/9.38 and DDIM-50 bare = 204.42. So the best composed
number in the paper is not the headline, the trend with steps runs *against* the usual expectation, the offered
explanation ("extrapolation error accumulating") is speculation not tested on the other rows (autoguidance w = 2 at
100 vs 200 steps is 11.33 vs 11.35 — no accumulation), and a model for which DDIM fails outright signals something odd
about the ε-parameterisation / alpha channel that a diffusion reviewer will probe.
(ii) `draft_experiments.md` §4.8.
(iii) **Bare CFG and autoguidance at 50 steps** (2 items ≈ 50 min; optionally label ref at 50 = 3 items) to show the
ordering is step-invariant and that 50-step improvement is shared, not method-specific; move the DDIM remark to the
appendix with a one-line hypothesis (ε-prediction at 16 px with hard alpha), and drop the "accumulation" sentence unless
it is tested.

### B.10 Length, redundancy, polish
(i) 10.5k words for ~6k of main text. Every headline number appears 3–5 times (21.98/8.59/7.67 in intro, method,
experiments, mechanism, limitations); §3.6 restates §5; §5.3 restates §4.6; §5.6 restates §4.9–4.10 *and* Limitations
¶2; §3.5 duplicates Table 10; Table 5 duplicates columns of Table 3. The intro has six contribution bullets, each
carrying a paragraph of numbers. TODO/TBD/"[TBD verify]" markers remain (param count, licence terms, figure wording, one
q16 cell). "Belief" is used as a term of art without definition until §3.2. Wording slips a reviewer will flag as
carelessness: "roughly halves" (A.1 #10), "within 1 %" (A.1 #7), "best CFG weight" (A.1 #11), "nearest" (A.1 #12).
(ii) Everywhere; see plan below.
(iii) The appendix plan.

#### Appendix plan (section → target words → what moves)

**Intro (1125 → 700).** Cut to four bullets: (1) rule + 16 px result, (2) composed + robustness under three metrics,
(3) directionality + characterisation, (4) cost and honest limitations (one sentence each on alignment cost and 12 px).
Delete the 20/24/32/v7s numeric restatement in bullet 2 ("Table 3–4" suffices), the negative-results bullet (one clause
inside bullet 3), the roadmap paragraph (line 20), and the DINOv2-patch explanation in ¶1 (→ §4.1, one sentence).

**Related (935 → 550).** Paragraph 1: one sentence + citation list; move the "we tested APG/FDG/… none improves" clause to
§5. Paragraph 2: keep, but add PAG/SEG. Paragraph 3 (SDXL): keep, tightened, and point to the bucketu number (B.3).
Paragraph 4 (multi-res): two sentences. Paragraph 5 (pixel art): halve; drop the q16 numbers (they are in §4). Paragraph 6
(evaluating tiny images): move to §4.1 protocol and appendix; remove the duplicated 7× and Spearman figures.

**Method (1550 → 1100).** Delete §3.5 Cost (Table 10 already carries it; leave one clause in §3.2). Move §3.6
"operational rule" to the appendix (or fold into §5's conclusion — it duplicates §5.1–5.5 point for point). Cut §3.4 to
three sentences (§2 already does this). Keep only one anchoring result per subsection (8.52 ± 0.29; 7.53 ± 0.19); the
36.43 "bad model" number and the snaplo paragraph go to §5.4 / appendix. Keep the two boxed equations and the b_low
paragraph (with the corrected statement about 24 px).

**Experiments (3851 → 2000).** Main text keeps: §4.1 protocol (trimmed), Table 1, Table 3 **with the reverse-control
column folded in** (delete Table 5), Table 4, condensed Table 7 (16 and 32 px rows only, or one line of text plus the
appendix table), condensed Table 9 (FID rows only). Appendix: Table 2 and §4.3 (keep three sentences: no CFG setting
beats w = 4 once B.1 is run; label w-curve flat vs snapshot steep; degraded-input references fail); Table 6 and §4.7
(→ §5.5 owns the probes); §4.8 sampler; Table 8 (keep one sentence + the bucketu operating point); Table 10 (one
sentence in §4.1); §4.12 qualitative → figure caption; the v7_lowres sentence; the 20 px sweep row once it has a source.

**Mechanism (1200 target, currently 2305).** Keep §5.1 with the 16 px table only (20/24 px table → appendix, replaced by
one sentence and `fig_tv_vs_fd_r`), §5.2 (tightened), §5.4 decomposition + timestep (drop the v7s decomposition sentence
— it is in §4.5), §5.5 probes (condensed; carries Table 6), and a two-sentence §5.7. Move §5.3 (higher buckets) into the
Table 3 discussion in §4 or appendix — it is the same content as §4.6. Move §5.6 to the appendix in full; its two
conclusions are already in §4.10 and Limitations ¶2. Add the interventional experiment of B.2 if run.

**Limitations (745 → 400).** ¶1 (alignment) to three sentences — the frontier detail lives in Appendix Table 8. Merge ¶2
(metric-specific) and ¶3 (floor/headroom) into one paragraph of four sentences. Keep ¶4 (12 px) as one sentence. ¶5
(scope) to three sentences; drop v7_lowres here (appendix). Broader impact → appendix or two sentences (ICLR does not
require it in the main body).

---

## Secondary notes (won't decide the score, will appear in reviews)

- Abstract says "explained by a controlled mechanism study" and "strictly directional"; B.2/B.4 argue for "characterised" and "directional at every resolution tested".
- `draft_intro.md:5` "More than half of the gap (a mean term of 12.13)": 12.13/18.53 = 65 % of the gap to the floor, or 55 % of raw FD — say which.
- After q16, autoguidance beats composed at 16 px (7.8 vs 8.3–8.4); Table 1 shows this in the last column but the prose sends it to an appendix ("we return to this in §4.9") while the appendix Table g′ is the one that actually addresses it — cross-reference correctly.
- Higher buckets at 16 px are *better than bare* in raw FD (17.14 vs 21.98) — the intro's "worse than no guidance" sentence lists 12/24/32 only to avoid this; state the 16 px exception explicitly rather than by omission.
- `draft_method.md:7` still has "[TBD: verify exact count]" for 72.5 M — an ICLR reviewer reading a TBD in Method will assume the rest is equally unverified. Resolve outline §5 item 17 before submission.
- The probe fine-tuning recipe (A.1 #2–4) is the only part of the paper with numbers the outline cannot vouch for; the mechanism section leans on those exact mean-term values (13.32 → 42.05; 8.1 vs 10.97 vs 3.25). Source them or remove them.
- "Belief" is evocative but undefined until §3.2; either define it in the intro on first use or use "lower-bucket prediction".
- Table 3 at 12 px gives 3-seed numbers while Table 5 gives seed 0 for the same two cells — see A.2; merging Table 5 into Table 3 removes the problem.

## Priority order if only ~6 GPU-hours are available before the deadline

1. CFG w ∈ {1, 1.5, 2, 3} + CLIP at 16 px (B.1) — 1.7 h. Decides whether −65 % survives matched alignment.
2. On-the-fly contrast-shrunk strong-prediction reference, 2 factors × 2 weights (B.2) — 1.7 h. Turns a correlation into an intervention.
3. PAG baseline, 2 weights (B.3) — 1 h.
4. Bare CFG + autoguidance at 50 steps (B.9) — 0.8 h.
5. probe_cg sampled statistics (B.2b) — 0.4 h.
6. Zero-GPU: caption split re-scoring (B.6), bootstrap CIs (B.7), third-extractor FD on saved samples (B.4), human study set-up (B.8), bucketu re-framing (B.3), appendix plan (B.10).
