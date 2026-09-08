# Review pass 2 — number audit and compression of `draft_full.md`

Scope: the **main body only** (everything before `## Appendix`, i.e. lines 1–236 of the final file) checked
number-by-number against `paper_assets/paper_outline.md` (Tables a, a′, a″, b, c, d, e, f, g, g′, g″, h, i, j, j′ and
the §3–§5 prose) and, where the outline is silent, against
`pixel_art_research_20260816/experiment_log.md`. Derived quantities (percentages, differences, sd multiples,
ratios) were recomputed rather than matched. ~430 numeric tokens in the main body were checked.

All line numbers below are **final** line numbers (after the compression of Part 2), unless marked *(pre-compression)*.

---

## Part 1 — Number audit

### (a) Numbers with no source in the outline or the log

| # | Value | Line | Status |
|---|---|---|---|
| a1 | **72,497,540** (parameter count, "72.5 M parameters (72,497,540 in the EMA state dict)") | 40 (§3.1) | **Unsourced.** It appears nowhere in `paper_outline.md` or `experiment_log.md`. The outline's gap list (§5 item 17) still says "Verify and record exact parameter count", and the log's last full status entry (09-08 12:25) still lists `72.5M 参数核实` as outstanding. The only record is the git commit message of `09b3e17` ("drafts: parameter count verified 72,497,540"). **Not changed** — but before submission the count needs a line in the log or the outline, otherwise Method carries the one number in the paper that cannot be traced. |

Nothing else in the main body is unsourced. In particular these three groups, which the outline does **not**
contain, were traced to the log and check out exactly:

- **Table 1, snaplo `+q16` seed 0 = 8.00** (line 100). Outline Table (a) still shows `TBD` for this cell;
  `[log §09-08 19:05]` gives `snaplo … q16 = 8.00 (复合 8.34, autog 7.82)`. ✓
- **§5.5 best-CFG simplicity statistics** (line 225): TV `26.2 / 27.9 / 29.0 / 30.6 / 32.0` at *w* = 1/1.5/2/3/4,
  ncol 73 → 68, PAG mid *w* = 2 ncol 68 / TV 29.1. `[log §09-08 19:05]` gives 26.20 / 27.92 / 29.00 / 30.55 /
  31.99, ncol 69/68/68/70/73, PAG 68 / 29.15, real TV 30.21. ✓ (cited in text)
- **§5.4 probe recipe and probe_cc statistics** (lines 213, 215), cited to `[log §09-07 13:11 / 19:05 / 19:35]` —
  verified in the earlier pass, not re-done here; the citations were preserved verbatim through the rewrite.

### (b) Numbers that disagree with the outline — both **fixed**

| # | Draft said | Source says | Fix |
|---|---|---|---|
| b1 | "the CLIP cost … is method-independent (snapshot and label references **within 0.2**)" — line 141 *(pre-compression)* | Outline text (paper_outline.md:340) also says "equal within 0.2", but outline **Table (g)** (paper_outline.md:331–335), reproduced verbatim as the draft's own Table 4, gives snapshot − label CLIP gaps of 0.21 / 0.11 / **0.26** / 0.20 at 16 / 20 / 24 / 32 px. The outline's prose contradicts the outline's table. | Changed to "**within 0.3**" (line 139). The table is unambiguous; 0.2 is wrong at 16 and 24 px. |
| b2 | Figure 3 caption enumerated **eight** points: 5 circles (bucket:12, snapshot, unconditional, bucket:24, bucket:64) + 3 crosses (block-average at *w* = 1.5 **and 2**, contrast-shrunk at *w* = 1.5) — line 203 *(pre-compression)* | Outline (paper_outline.md:483) and `autonomy_state.md` both describe `fig_tv_vs_fd.png` as a **7-point** plot; `draft_mechanism.md:39` says "the seven 16 px references … (the five above plus the **two** trained branches)". | Caption now reads "`fig_tv_vs_fd.png`, **seven points**" with "the **two** trained degraded-view branches … both at $w = 1.5$" (line 201). |

### (c) Internal contradictions inside `draft_full.md`

**Fixed (the paper's own table settles them):**

| # | Contradiction | Fix |
|---|---|---|
| c1 | §5.5 claimed the guided rows are "the only ones that reduce the colour count towards real … while **keeping TV at or above the real value**". The paper's own Table 6 (lines 184–195) gives guided TV **29.0** (autoguidance) / 32.4 (label) / **30.1** (composed) against real **30.2** — two of the three are *below* real, so the sentence contradicts the table it cites. `[log §09-08 19:05]` says `TV 保持在真实值附近或以上` ("at or **near** the real value or above"). | Line 225 now reads "holding TV **at or near** the real value (**29.0–32.4 against 30.2**)". |
| c2 | §2 said the SDXL analogue "at $w = 2$ [is] **no better than** the best CFG point (12.24)", while the value it quotes for that row is **12.10** — i.e. marginally *lower* than 12.24 (and well inside the 0.2–0.5 seed sd). | Line 32 now reads "and at best **level with** the best CFG point (12.24)". |

**Listed, not changed** (each needs an author decision the outline cannot settle):

| # | Contradiction | Detail |
|---|---|---|
| c3 | **§4.7 overstates the Inception result on v7s.** Line 159 says "the composed reference is best or tied-best at every resolution **and on both models**, within seed spread of autoguidance at 16 px". The paper's own Table 5 (v7s @16 row) gives autoguidance **8.25** and composed **8.49** — autoguidance is ahead by 0.24 FID, and the qualifier ("within seed spread at 16 px") is attached only to the v7h row. Appendix A.10 (line 398) repeats the same unqualified claim. The outline makes the identical claim (Table (h) reading (i)), so it is the outline *and* the draft that overstate; the table is the honest source. Suggested wording: "best or tied-best at every resolution on v7h and at 20 / 24 px on v7s, with autoguidance 0.24 FID ahead at 16 px on v7s". |
| c4 | **Two different "real 16 px" statistics.** Table 6 (line 186) gives real 16 px sprites `ncol 34, flat .202, TV 30.2`, while §5.1 (line 180) quotes "real 12 px sprites: 5 colours, flat ≈ .51" from outline Table (d)'s separate *Real native 12 px / 16 px* row, whose 16 px entry is `ncol 6, flat .42, TV 30`. Both rows exist in the outline, presumably matched `to_tensor(16)` preprocessing vs the native files, but the paper never says so, and a reader sees the same population with 34 and 6 median colours. One clarifying clause in the Table 6 caption would close it; the outline does not state which preprocessing each row used, so nothing was changed. |
| c5 | **Mixed seed conventions for the same cell.** Bare CFG @20 px appears as `47.04 ± 1.11` (Table 2) and `45.92` (Tables A1, A7); the composed row @16 px as `7.53 ± 0.19` (Table 1) and `7.67` (all of §5 and Tables 3, 5, 6). Every occurrence is labelled with its seed basis and each value is correct, so this is presentation, not error — but it is the same issue `review_pass1.md` §A.2 raised, and it survives into the merged draft. |
| c6 | **Derived numbers removed by compression.** The best-CFG-to-floor gaps `27.4 / 54.8 / 69.9 vs 9.0` (§4.4, pre-compression) are gone; they are recomputable from Table 2 (39.57 − 12.14 etc.), and §6 keeps the qualitative form ("3–7× above the floor"). Noted so the author knows the sentence was dropped, not mis-stated. |

### Checked and clean (so nothing is re-litigated)

Every value of Tables 1–6 reproduces outline Tables (a), (b)/(j′), (c), (g)/(g″)/(j′), (h) and (d) exactly.
Every derived quantity was recomputed and matches: composed vs best-CFG **−40 / −27 / −29 / −17 %** and vs *w* = 4
**−65 / −39 / −39 / −28 %**; label **−32 / −17 / −12 / −7 %**; autoguidance **−30 / −20 / −20 / −10 %** and
**−12 %** @12; re-tuning gains **−42 %** @16 and **−16 / −14 / −14 %** @20/24/32; v7s **−45 / −35 / −36 %**,
**−62 / −43 / −39 %**, label **−33 / −19 / −17 %**, autoguidance **−27 / −30 / −29 %**, re-tuning
**−32 / −11 / −6 %**; Inception **−8 / −10 / −14 / −11 %**; over-guidance **72 %** (3-seed) ; sd multiples
**6.9 / 15.5 / 25** and **4.6 / 3.9 / 8.2**; headroom **6.5×**; bare-to-floor **3–7×** / **3–6×**; ncol ratio
**1.8×** and flat "a quarter"; CLIP deltas **0.32**, **+9.7 FD**, **≈ +3 FD per 0.1 CLIP**, **0.54 / 0.56 / 0.65**,
**0.16 / 0.21 / 0.47**, **30 / 30 / 29 %**, **+0.9 FD**; q16 margins **−20 / −20 / −8 %**.

A machine diff of every numeric token in the main body before and after the compression returns **zero new
numbers** — no value was altered, rounded or invented while cutting.

---

## Part 2 — Word counts (whitespace tokens), before and after

Prose excludes table rows, table/figure captions and headings, which are counted separately.
"Before" = the draft as received (`0839d5d`); "After" = the current file.

| Section | prose before | prose after | tables | captions before → after | headings | section total before → after |
|---|---|---|---|---|---|---|
| Title | — | — | — | — | 14 | 14 → 14 |
| Abstract | 215 | **206** | 0 | 0 → 0 | 2 | 217 → **208** |
| 1 Introduction | 841 | **686** | 0 | 0 → 0 | 3 | 844 → **689** |
| 2 Related Work | 612 | **504** | 0 | 0 → 0 | 4 | 616 → **508** |
| 3 Method (3.1 / 3.2 / 3.3) | 815 (312/307/196) | **670** (266/264/140) | 0 | 0 → 0 | 15 | 830 → **685** |
| 4.1 Setup | 319 | **238** | 0 | 0 → 0 | 3 | 322 → **241** |
| 4.2 Main results at 16 px | 135 | **125** | 229 | 155 → 155 | 7 | 526 → **516** |
| 4.3 Weight sweeps / zero-training baselines | 241 | **147** | 0 | 0 → 0 | 7 | 248 → **154** |
| 4.4 Resolutions and reverse controls | 386 | **302** | 296 | 88 → 88 | 8 | 778 → **694** |
| 4.5 Second model | 171 | **158** | 149 | 55 → 55 | 7 | 382 → **369** |
| 4.6 Alignment–fidelity frontier | 543 | **395** | 258 | 143 → 143 | 4 | 948 → **800** |
| 4.7 Second metric family / quantisation | 396 | **317** | 167 | 49 → 49 | 8 | 620 → **541** |
| **§4 subtotal** | **2,191** | **1,682** | **1,099** | **490 → 490** | **47** | **3,827 → 3,318** |
| 5 (lead-in) | 87 | **73** | 0 | 0 → 0 | 12 | 99 → **85** |
| 5.1 What the belief looks like | 200 | **155** | 220 | 62 → 62 | 8 | 490 → **445** |
| 5.2 TV predicts the guided FD | 237 | **151** | 0 | 80 → 79 | 11 | 328 → **241** |
| 5.3 What each reference corrects | 351 | **234** | 0 | 0 → 0 | 6 | 357 → **240** |
| 5.4 Controlled references | 620 | **463** | 0 | 0 → 0 | 18 | 638 → **481** |
| 5.5 What remains | 258 | **244** | 0 | 0 → 0 | 4 | 262 → **248** |
| **§5 subtotal** | **1,753** | **1,320** | **220** | **142 → 141** | **59** | **2,174 → 1,740** |
| 6 Limitations and Conclusion | 488 | **471** | 0 | 0 → 0 | 5 | 493 → **476** |
| **MAIN BODY** | **6,915** | **5,539** | **1,319** | **632 → 631** | **149** | **9,015 → 7,638** |
| Appendix | 2,771 | 2,771 | 2,662 | 645 → 658 | 143 | 6,221 → 6,234 |

**Headline figures.** Main body prose + tables + captions: **8,866 → 7,489** (−15.5 %); including headings and
title, **9,015 → 7,638**. Prose alone: **6,915 → 5,539** (−19.9 %).

This lands ~7 % above the ~7,000 target rather than on it. The residue is not padding: with all six main tables
(1,319 words) and three figures retained, every headline number kept in prose, and the honest-framing sentences
kept verbatim, the remaining paragraphs average one sourced number per 8–10 words. Reaching 7,000 exactly would
mean dropping a main table (Table 5 and Table 6 are the candidates, both fully duplicated by Appendix A8 / A7) or
demoting the per-resolution percentage triples in §4.4 / §4.5 / §4.7 to the appendix — both are author decisions,
not editorial ones, so they were left alone.

---

## Part 3 — What was cut, and where it went

Nothing was deleted outright: every removed value already had, or was given, a home in the existing appendix
sections. No appendix section was created; three were extended.

| Cut from | What | Where it lives now |
|---|---|---|
| §3.3 | The whole **Cost** paragraph (72.7 vs 73.8 s, 4951 MiB, 73.3–74.0 s) | Already in **A.11** / Table A9 verbatim; §3.3 keeps the claim plus the `+556 MiB` figure and a pointer |
| §4.1 | The enumerated **controls list** (CADS, interval CFG, PAG, reverse, `bucketmix`, `probe_cg`/`probe_cc`, `shrink:f`) and the long `+q16` definition | Each control is now defined where it is used (§4.3, §4.4, §5.4); q16 → **A.9** |
| §4.3 | The full 8-point **CFG weight curve**; the autoguidance snapshot-step series (14.89 / 11.33 / 12.14 / 11.62); the "(iii)" sub-paragraph | **A.1** Table A1 (both series verbatim) |
| §4.4 | Reverse-control **precision/recall values** (.935, .85–.88, .907/.902); best-CFG-to-floor gaps | **A.14** already carries the P/R numbers; the gaps are recomputable from Table 2 (see c6) |
| §4.6 | The **CLIP-vs-*w* curve** (29.43 / 29.74 / 29.87 / 30.01 / 30.06); "3–5 FD below the best CFG point"; the per-resolution `bucketu` FD/CLIP pairs at 20/24 px | **A.2** Table A2 (curve, mean/cov, R@1) and **A.3** (all `bucketu` rows at 12–32 px, Figure A1) |
| §4.7 | Inception **per-resolution floors** (4.93 / 6.72 / 8.23 / 8.99 / 9.75); the five Spearman values; PAG's KID | Table **A8** caption (floors) and **A.10** (Spearman list, PAG 8.92 / 1.31) |
| §5.1 | Sample count in the protocol clause; the "anti-correlated own quality" sentence | §4.1 (protocol) and **A.8** |
| §5.2 | The long restatement of the structure-alignment clause (bucket:12 @20 px, opacity .455 vs .362) | **A.8** paragraph and Table A7 |
| §5.3 | The mean-term curve (9.47 / 5.87 / 6.32 / 8.70 / 12.13); interval-scheduling mean/cov pairs; FDG's 2.5 → 3.54 → 4.8 series | **A.2** (mean/cov per *w*), **A.10** / **A.13** (interval and FDG rows) |
| §5.4 | The probe fine-tuning detail ("class embedding 7 → 14"); the `probe_cg` / `probe_cc` weight sweeps (35.22 / 59.38, 19.98 / 44.90); the `shrink:f` CLIP range | **A.6** (recipe and mean terms), **Table 6** (kept in the main body, carries both sweeps), and **A.5** caption — extended with "CLIP 28.5–29.5 at R@1 5–12 %" so the interventional control's alignment numbers are not lost |
| §1 / §2 / §6 | Repetition only: the §1 bullets no longer restate the §4.4 / §4.6 / §5 numbers a second time; §2 ¶2 and ¶4 lost connective prose (every `[cite:` / `[cite?` marker is intact); §6 lost duplicated clauses already stated in §4.6 / §4.7 | — (repetition, nothing to relocate) |

**Preserved deliberately, as instructed:** all six main tables and three figures; every headline number; and the
honest-framing sentences — best-CFG as the baseline throughout (22 occurrences), "Inception margins … smaller but
**of the same sign** at every resolution" (§4.7), "a **characterisation, not** a mechanism" (§5.2) and "Section 5 is
a characterisation plus one intervention, **not a proven mechanism**" (§6), PAG "tracks the CFG optimum without
removing the systematic error" (§4.3) / "lies **on the CFG curve**, not the guided frontier" (§4.6) /
"**indistinguishable from CFG at the same weight**" (§5.5), the q16 order **swap** at 16 px (§4.2, §6), and the
per-resolution CLIP signs — CFG *w* = 4 "**over-aligns**" (30.06 vs 29.80) while the guided rows sit ≈ 0.3 below
real at 16 px and **0.16 / 0.21 / 0.47** below real at 20 / 24 / 32 px.

All five `[TODO cite …]` bibliography markers survive (`CLIP`, `Ho et al. 2020`, `BLIP`, `clean-fid`, `OGA`), as do
the three `[log §…]` citations in §5.4 and §5.5. The one remaining `[TBD]` in the file is inside
`[TODO cite OGA; TBD: state licence terms]` in Appendix A.17 — pre-existing and outside the main body.

---

## One item outside the audit scope, worth an author minute

Appendix Table A6 (caption line 356; best-CFG row line 362) still shows `—` for the best-CFG row's precision / recall / density / coverage,
although `[log §09-08 19:05]` records them (best-CFG *w* = 1.5 seeds 0/1/2 = 12.24 / 13.32 / 11.91, mean term
5.87 / 6.64 / 5.78, precision .931 / .920 / .928, recall .886 / .897 / .890) and the log's own note says
"表 A6 无 TBD". The composed row's seed-2 values (precision .925, recall .900, density .954, coverage .924) are
likewise unrecorded in A6.
