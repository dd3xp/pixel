# Read-through pass 3 — logical audit of `draft_full.md`

Scope: logic only. Numbers were verified by `review_pass1.md` / `review_pass2.md` and references by
`citation_audit.md`; nothing here is an arithmetic re-check, and where a number appears it is used only to show that
two passages cannot both be true. No file was edited.

**Line numbers** refer to a frozen snapshot of `draft_full.md` taken at 2026-09-09 06:37 (md5
`24a206d886401beac61c34942eb1fa2d`, 478 lines). The file was being edited by another process during this pass — it
changed twice while I read it (both times inside A.10 / Table A8). Everything at or below line 396 is unchanged from
the version I first read; appendix lines above 396 may drift by one or two. Every item also carries a quote, so the
anchors survive the drift.

---

## Critical

### C1. The TV comparison — the paper's entire explanatory layer — is made across two different sampling regimes
**L15** ("TV 21.6 against 32.0 for the strong prediction"), **L180**, **L182 Table 6**, **L200**, **L466 (A.15 rule 1)**,
**L391 (Figure A6)**.

Every weak reference in Table 6 and Table A7 is sampled with `--cfg 0` (caption L182: "sampled with `--cfg 0`"), but
the row it is compared against, "Strong model (v7h, CFG $w=4$)", is sampled at guidance weight 4. The paper's own
§5.5 (L226) shows that TV is strongly driven by the guidance weight: "26.2 / 27.9 / 29.0 / 30.6 / 32.0 at
$w = 1 / 1.5 / 2 / 3 / 4$". The regime-matched comparator for a `--cfg 0` belief is the $w = 1$ row — the conditional
prediction under the correct label, followed alone — i.e. **TV 26.2, not 32.0**.

Three consequences, all load-bearing:

1. The headline contrast deficit shrinks from 21.6 vs 32.0 (−33 %) to 21.6 vs 26.2 (−18 %).
2. The snapshot reference has TV 27.3, i.e. **above** the matched comparator, yet it is one of the two references
   that work. Under the matched normalisation its $x = \mathrm{TV_{ref}}/\mathrm{TV_{strong}} = 1.04 \ge 1$ with
   $y < 1$, which directly falsifies the rule stated at L200 and L391 ("every one with $x \ge 1$ gives $y \approx 1$
   or $y > 1$") and A.15 rule 1 ("Lower TV than the strong model is necessary").
3. The effective/ineffective boundary at 20 / 24 px is decided by 1–2.5 TV units (bucket:24 28.5 vs strong 31.0;
   bucket:32 31.3 vs strong 30.3, L383–L386) — smaller than the ~6-unit artefact the $w = 4$ comparator introduces
   at 16 px, and the unguided comparator at 20 / 24 px was never measured at all.

*Suggested rewrite.* Add the regime-matched row: sample the correct-label belief at `--cfg 0` at each resolution
(three cheap runs) and use its TV as $\mathrm{TV_{strong}}$; report the $w = 4$ value only as context. Then restate
the rule honestly: "Among same-caption beliefs the ranking by TV tracks the ranking by guided FD; the snapshot,
whose TV is at the level of the unguided conditional belief, is the exception that shows TV is not the whole story
(§5.2)." If the runs are not affordable, say in §5.1 that $\mathrm{TV_{strong}}$ is measured on CFG $w=4$ samples
while references are measured unguided, and that the comparison is therefore an upper bound on the deficit.

### C2. The stated failure mode ("insufficient local contrast") is contradicted by the paper's own contrast statistic
**L7** ("regress towards the mean — bleeding colours, insufficient local contrast"), **L15** ("lower contrast is
exactly the direction of the error"), **L102** ("The guided rows reduce the bare model's bleeding colours and
insufficient local contrast").

Table 6 (L186–L188) gives real sprites TV 30.2 and the bare model TV **32.0** — the model has *more* local contrast
than the data, not less. And the guided rows *lower* TV rather than raising it: autoguidance 29.0, composed 30.1
(L196). So the correction does not add missing contrast; §5.5 (L224) admits as much for the label row ("does not
lower the TV of the *guided* samples"), and pass 2's fix c1 already softened §5.5 without touching the abstract, the
intro or Figure 1. What the guided rows actually repair is colour fragmentation: ncol 73 → 60.5 against 34 real,
flat .030 → .044 against .202.

*Suggested rewrite.* L7: "…regress towards the mean — bleeding colours, roughly twice the real number of distinct
colours per sprite and almost no flat regions". L15: keep "systematically lower local contrast" as a description of
the *reference* and delete "lower contrast is exactly the direction of the error"; replace with "and extrapolating
away from it removes the colour-mean shift that CFG cannot reach (§5.3)". L102: "The guided rows reduce the bare
model's colour bleeding (median unique colours 73 → 60–64 against 34 real)".

### C3. The one interventional control does not exclude the obvious alternative explanation
**L218** ("$f = 0.85 / 0.7 / 0.5$ at $w = 2$ give 74.8 / 200.4 / 374.6 … The belief's difference from the strong
prediction is therefore *spatially structured* … not a global scalar"), restated at **L24**, **L7** and **L236**.

With $e_\text{ref}$ = the strong $\hat{x}_0$ shrunk by $f$ about its mean, the update is exactly
$\hat{x}_0 \mapsto m + [f + w(1-f)](\hat{x}_0 - m)$: a **fixed multiplicative gain applied at every one of 100
steps** — 1.15× per step at $f = 0.85, w = 2$; 1.5× at $f = 0.5$. Divergence is the expected behaviour of any
sustained fixed gain, whatever its spatial structure, and the numbers show exactly that signature (monotone
explosion, and $f \to 1$ degenerating to no guidance at 16.39). The lower-bucket reference's implied gain is neither
fixed nor uniform across $t$, so the two are not matched in effective guidance strength. The paper's conclusion —
that the difference must be spatially structured — is one reading; "the shrink control was run at a much larger
effective gain" is another, and nothing in the draft separates them. This is the paper's only intervention, and
§5.4's title and abstract sentence both rest on it.

*Suggested rewrite (needs one cheap run, or an honest downgrade).* Either (a) gain-match: report
$\lVert e_\text{strong} - e_\text{ref}\rVert$ for `bucket:12` and for `shrink:f`, and re-run `shrink` at the $w$ that
matches it (e.g. $f = 0.5$ with $w \approx 1.1$, giving a per-step factor ≈ 1.05); or (b) downgrade the claim to
"a *uniform, fixed-gain* contrast shrink of the strong prediction does not reproduce the label's effect — it
diverges. We cannot separate 'the difference is spatially structured' from 'the shrink applies a much larger
effective gain per step', so this control rules out only the simplest scalar reading."

### C4. The title's superlative is contradicted by Table 2
**L3** ("A Diffusion Model's Lower-Resolution Belief Is Its Own Best Bad Model").

The lower-resolution belief is not the best bad model available to this model. As a single reference it *loses* to
the stored 10 k snapshot at 20 / 24 / 32 px (Table 2, L120–L122: 32.75 vs 31.57; 59.32 vs 54.15; 77.45 vs 75.23) and
ties it at 16 px (8.52 ± 0.29 vs 8.73 ± 0.26, explicitly "within seed spread", L88). Only the *composition* wins,
and the composition is not free — it stores the snapshot. The title therefore promises the one thing §4.4 and §6
retract.

*Suggested rewrite.* "Cross-Resolution Self-Guidance: A Bucketed Diffusion Model Contains Its Own Weak Reference"
— or keep the belief-as-bad-model framing but drop "Best".

### C5. "at every resolution" in the conclusion contradicts §6 and §3.2
**L236** ("lowers FD-DINOv2 by 17–45 % below the best CFG weight at every resolution and on both models tested").

$R \in \{12,16,20,24,32\}$ (L40), and at 12 px the method does not exist: L66 "At 12 px no lower bucket exists and
the method does not apply"; L234 "at 12 px only autoguidance applies". The conclusion is the last thing a reviewer
reads.

*Suggested rewrite.* "…at every resolution that has a lower bucket (16–32 px), on both models tested; at 12 px, the
smallest bucket, only the snapshot reference is available."

---

## Major

### M1. The abstract borrows one row's CLIP to decorate another row's FD
**L7** ("composing this reference with an early snapshot lowers FD-DINOv2 by 40 / 27 / 29 / 20 % … at equal or better
CLIP alignment than the CFG optimum at 16 px").

The composed row scores CLIP 29.51 against best-CFG's 29.74 at 16 px (Table 4, L152) — worse, not "equal or better".
The row that reaches 29.77 is composed∘`bucketu:12`, whose FD is 8.60, i.e. −30 % not −40 %. As written the abstract
claims a Pareto point the paper does not have.

*Suggested rewrite.* "…at the CLIP level of CFG's own FD optimum (29.51 vs 29.74 at 16 px, both ≈ 0.3 below real);
folding the empty caption into the weak branch restores real-data-level alignment (29.77) at +0.9 FD."

### M2. PAG is dismissed by a comparison standard the paper refuses to apply to itself
**L30** ("PAG … tracks the CFG curve at every weight"), **L106** ("a gentler CFG"), **L141** ("PAG lies on the CFG
curve, not the guided frontier"), **L7** ("perturbed-attention guidance matches CFG at every weight").

§4.6 and A.3 establish that the right comparison is at matched CLIP: "Against the CFG weight of equal alignment …
those rows are 30 / 30 / 29 % lower in FD" (L143). Apply that standard to PAG using the paper's own Table A2: PAG
`pag:mid` $w = 1.5$ is (12.57, CLIP 29.45); the CFG curve at CLIP 29.43 is $w = 1$ = **16.39**, and A.3 itself says
"the CFG curve passes at FD ≈ 14–15 for CLIP 29.5–29.6" (L306). PAG is therefore ~20–23 % below the CFG curve *at its
own alignment* — a smaller version of the paper's own claim, not "on the curve". Separately, "at every weight" comes
from three weights at one resolution.

*Suggested rewrite.* "PAG shifts the frontier in the same direction but far less: at its own alignment (CLIP 29.45)
it is ~20 % below the CFG curve, against ~40 % for the composed reference, and its FD never falls below 12.5 at any
weight we tested ($w = 1.5 / 2 / 3$, 16 px)."

### M3. Contribution 3 claims a frontier result over a range the guided rows never reach
**L23** ("the guided rows lie below the CFG curve at every alignment level it reaches").

The CFG curve reaches CLIP 30.06; §4.6 states twice that no guided variant does — "None of the seven variants reaches
the $w = 4$ CLIP of 30.06" (L141, repeated L304). Figure 2's caption gets it right ("in the 29.4–29.9 range", L145);
the contribution bullet does not.

*Suggested rewrite.* "…the guided rows lie below the CFG curve at every alignment level they reach (CLIP 29.4–29.8);
no guided variant matches CFG's maximum alignment of 30.06."

### M4. "The gain is largest where the bare model is furthest from the floor" is backwards
**L110** (and **L234**, "the gain is largest where the model is weakest").

Composed gains are −40 / −27 / −29 / −20 % at 16 / 20 / 24 / 32 px, while the bare-to-floor ratio is 6.4 / 3.8 / 6.2 /
7.0 and the absolute gap 18.5 / 33.8 / 66.8 / 83.1. On either reading, 32 px is the furthest from the floor and has
the *smallest* gain; 16 px is not the furthest and has the largest. The claim is used in two places to explain the
resolution trend, and it explains it in the wrong direction.

*Suggested rewrite.* "The gain is largest at 16 px — the resolution that dominates the training population (§3.1) —
and decreases with $R$, even though the bare model's distance from the floor grows; we have no account of why, and
the label-only component tracks the same trend (§4.4)."

### M5. §3.3 says the composed gain does not shrink with resolution; Table 2 says it does
**L76** ("its stand-alone gain shrinks with resolution whereas the composed gain does not (§4.4)").

Table 2: composed −40 / −27 / −29 / −20 %. It shrinks by half from 16 to 32 px. The true contrast is that it shrinks
*less* than the label-only gain (−32 → −8 %).

*Suggested rewrite.* "…its stand-alone gain shrinks steeply with resolution (−32 % → −8 %) while the composed gain
shrinks much less (−40 % → −20 %)".

### M6. §5.4's "flips the sign" result is measured against the baseline the paper elsewhere rejects
**L216** ("`probe_cc` … is *effective*, 14.71 against 20.63 bare"), **L344** ("no CFG weight sweep was run on the
fine-tuned weights, so these rows are compared within the table only").

"Bare" here is CFG $w = 4$ on the fine-tuned weights, and the paper's central methodological point is that $w = 4$ is
over-guided by ~40 % at 16 px (L106). A re-tuned CFG on the probe weights would plausibly land near 12–13 — below
probe_cc's 14.71 — in which case probe_cc is not "effective" at all under the paper's own standard, and the sentence
"a structure-aligned degradation flips the sign" loses its evidence. The appendix admits the sweep is missing; §5.4
uses the word anyway.

*Suggested rewrite.* "…lowers FD against the $w = 4$ default on the same weights (14.71 vs 20.63), reversing
`probe_cg`'s sign. We did not sweep CFG on the fine-tuned weights, so we cannot say whether it also beats a re-tuned
CFG there; on the same weights the free label reference reaches 9.43."

### M7. The promised selection rule cannot decide the only case where the default fails
**L66** ("selection rule in Appendix A.15"), **L462**, **L466**.

The one non-obvious choice in the paper is 24 px, where the nearest lower bucket (20) is nearly useless (66.20 vs
best-CFG 67.34) and bucket:16 is used instead. A.15's rule is a TV rule — but the TV of the bucket:20 belief at 24 px
was **never measured**: Table A7 lists only bucket:16 and bucket:32 at 24 px (L385–L386). The same holds for
bucket:16 at 32 px (91.26, worse than best-CFG 84.63, no TV). So the rule has been demonstrated only on cases whose
answer was already known, and the 24 px choice was in fact made on the reported test FD (acknowledged, L234). A
section that promises an operational rule and delivers a post-hoc rationalisation is exactly what §6's honesty
cannot cover.

*Suggested rewrite.* Either measure the two missing beliefs (`--cfg 0`, 2 runs) and show whether the rule predicts
bk20@24 and bk16@32, or retitle A.15 "Which references work, and which do not" and state plainly: "The TV statistic
separates references that help from references that hurt; it does not choose among lower buckets, and our 24 px
choice was made on the reported FD."

### M8. The TV rule is validated against the over-guided default, and its point set excludes the awkward cases
**L200** ("every one of the ten bucket and snapshot references … with $x < 1$ gives $y < 1$"), **L391**
($y = \mathrm{FD_{guided}}/\mathrm{FD_{bare}}$, with bare = CFG $w = 4$).

Beating the $w = 4$ default is a low bar the paper spends §4.3 arguing against. Against best-CFG the rule already has
counterexamples inside the draft: bucket:20 at 24 px (66.20 vs best-CFG 67.34 — $y < 1$, no real gain) and
bucket:16 at 32 px (91.26 vs 84.63 — $y < 1$, clearly worse than the baseline). Neither is among the ten points, and
they are precisely the two lower-bucket references whose FD lands between best-CFG and bare.

*Suggested rewrite.* State the denominator explicitly and add the caveat: "Normalised by the $w=4$ default, every
low-TV same-caption reference improves; against re-tuned CFG the rule is weaker — two low-TV references (bk20 at
24 px, bk16 at 32 px) fail to beat best-CFG. TV separates helpful from harmful, not good from best."

---

## Moderate

### Mo1. Contribution 4 says every helping reference keeps the strong prediction's colour count and structure; Table 6 and Table A7 say otherwise
**L24** ("Every reference that helps has lower TV than the strong prediction while keeping its colour count and
structure").

The snapshot helps (8.98) and has ncol 95 vs 73 and flat .000 vs .030 (L190). At 20 px the bucket:12 belief helps and
Table A7 annotates it "opaque .455 ≫ .362: structure drift" (L381); at 24 px the effective bucket:16 belief has
opaque .425 vs .363 and ncol 142 vs 130.

*Suggested rewrite.* "Every reference that helps has lower TV than the strong prediction and follows the same
denoising trajectory; colour count and opaque area are closely matched for the label reference at 16 px and drift at
larger $R$ and for the snapshot."

### Mo2. "Structure-aligned by construction" claims more than "only the label differs" licenses
**L64**. Changing the label changes the prediction, including where opacity falls (Table A7's own "structure drift"
annotation). *Rewrite:* "structure-aligned in practice — the same $x_t$, caption and trajectory, and the same layout
in every sample we inspected — though the opaque area drifts at 20 / 24 px (.455 vs .362 at 20 px)."

### Mo3. Only one of the "two variants that dominate" actually dominates
**L141** ("two zero-training variants *dominate* the best CFG point (12.24, 29.74) in both coordinates … composed
plus an additive CFG term (`cfg_text` 1.5, 3 NFE) at 9.19 / 29.72").

29.72 < 29.74, so the second variant does not dominate; it is also 3 NFE, i.e. not cost-matched with the point it is
said to dominate. *Rewrite:* "one zero-training variant dominates the best CFG point in both coordinates
(composed∘`bucketu:12`, 8.60 / 29.77, same 2 NFE); a 3-NFE variant reaches 9.19 at essentially the same alignment
(29.72 vs 29.74)."

### Mo4. Two incompatible descriptions of what the composition inherits
**L206** ("keeps the label's mean term (2.40) and most of the snapshot's covariance gain") vs **L369** ("the
composition takes roughly half of each reference's benefit"). The numbers (2.40 vs 2.55 / 3.88; 5.27 vs 5.09 / 6.04)
support the first. *Rewrite:* delete the "roughly half of each" sentence in A.7 or align it with §5.3.

### Mo5. A forward reference points at the wrong table and the wrong configuration
**L210** ("channel-decoupled weights show FD is set almost entirely by the RGB weight", cited to "Appendix Table
A10", in a paragraph about "second components … on top of the composed reference").

The channel-decoupled rows are in Table A1 (L272 region) and A.13's prose, and A.13 labels them "(single-reference
autoguidance)" — they were never run on the composed reference. *Rewrite:* "…and, on single-reference autoguidance,
channel-decoupled weights show FD is set almost entirely by the RGB weight (Appendix A.13)."

### Mo6. The intro attributes to PAG a decomposition result that was never computed
**L13** ("neither CFG variants nor perturbed-attention guidance … repair it", where "it" is the feature-mean shift).

Table A2 shows "—" in the mean/cov column for every PAG row (L293). The claim is inferred from total FD alone.
*Rewrite:* "…and no CFG variant we tried, nor perturbed-attention guidance, brings total FD below 12.5 (§4.3); we
decomposed FD only for the rows in Table A2."

### Mo7. "The composed row still wins at every resolution" after quantisation is contradicted twice, and the q16 comparison against best-CFG does not exist above 16 px
**L161** ("although the composed row still wins at every resolution afterwards"), against **L88** and **L232**
("after q16 autoguidance beats the composed row at 16 px") and **L395** (A.9 quantises only the $w = 4$ default and
the guided rows at 20 / 24 / 32 px; best-CFG is quantised at 16 px only, 11.55).

This also over-reaches at **L232**: "The composed reference and the reverse-control result hold under every metric
and against best-CFG at every resolution" — under the q16 metric, the best-CFG comparison at 20 / 24 / 32 px was
never run. *Rewrite (L161):* "so although the composed row still beats the $w = 4$ default at every resolution after
quantisation — autoguidance overtakes it at 16 px — its margin shrinks from …; we did not quantise the best-CFG
samples above 16 px." And at L232, qualify "under every metric" to "under DINOv2 and Inception at every resolution,
and under q16 against the training default".

### Mo8. The three-point TV ordering in §5.4 mixes a target statistic, a branch statistic and a belief statistic across three weight sets
**L220** ("Three references with TV 14.2 / 15.9 / 21.6 thus span harmful / weakly effective / effective, in reverse
order of TV").

14.2 is `probe_cg`'s **training target**, not its sampled branch — Table 6's own caption says the branch "was not
measured" (L182) — and Figure 3 (L202) plots it on an axis labelled "TV of the pure weak reference". 15.9 is a
sampled branch on `probe_cc`'s fine-tuned weights; 21.6 is a belief on v7h. The three also have three different bare
baselines (19.17 / 20.63 / 21.98). *Rewrite:* measure `probe_cg`'s branch (one `--cfg 0` run, already recommended in
pass 1), or write "…in reverse order of the TV of their degraded views (block-average target 14.2, contrast-shrunk
branch 15.9) and of the belief itself (21.6), on three different weight sets".

### Mo9. "Bare", "no guidance" and "strong model/prediction" are three overlapping names for two or three things
**L52** ($e_\text{strong}$ = the conditional $\epsilon$ at $b_R$, no CFG) vs **L176** ("'strong model' means the CFG
$w = 4$ samples") vs **L110 / L159 / L163** ("bare" = CFG $w = 4$) vs **L218** ("16.39 for no guidance at all",
i.e. $w = 1$) and **L412** (Table A9 calls $w = 1$ "No guidance"). The TV and FD normalisations of §5.2 and Figure A6
depend on which of these is meant, and item C1 above is a direct consequence.

*Rewrite:* define the three terms once in §4.1 — "*unguided* = $w = 1$ (the conditional prediction followed alone);
*bare* = the training default $w = 4$; the *strong prediction* $e_\text{strong}$ = the conditional $\epsilon$ under
$b_R$" — and use them consistently, including in the axis labels of Figures 3 and A6.

### Mo10. Terms used before they are defined, and one row with two names
- `bucketu` first appears at **L32** (Related Work) and **L102** (Figure 1); it is defined only at **L141** and
  **L304**.
- `bucketmix` is used at **L112** ("`bucketmix:12` (9.42) lies between…") and expanded only inside a Table A1 row
  label (L269) and A.14.
- "*matched*" Fréchet distance is italicised at **L82** as if being defined, and never is; it recurs in the caption
  of every table.
- **L76** names the row "snaplo"; Table 1 calls it "Composed, snapshot only for $t/T\in[0,0.5]$" (L100) and Table A10
  "Interval scheduling of snapshot ref" (L438 region). Three names, one row.

*Rewrite:* gloss `bucketu` and `bucketmix` at first use ("`bucketu:b` = the weak branch takes the lower bucket *and*
the empty caption"), define "matched" (identical preprocessing and sample count for real and generated sets) or drop
the italics, and pick one name for the interval-restricted row.

---

## Minor

- **m1. L82** — "Seed-to-seed sd is 0.2–0.5 FD at 16 px, and differences of that order are not interpreted", but
  Table 1's own best-CFG row is ± 0.72 (L96) and Table 2 has ± 1.09 at 32 px and ± 1.37 for bk16@24 px. Meanwhile
  differences of 0.13 are interpreted ("best or tied-best", Inception 7.97 vs 7.84, L159). *Rewrite:* "0.2–0.7 FD at
  16 px, rising to ~1.6 at 32 px".
- **m2. L110 / L159 / L236 vs L7 and Table 2** — the 32 px composed gain is quoted as −17 % in §4.4, §4.7 and the
  conclusion's "17–45 %", and as −20 % in Table 2 and the abstract. Both are correct on different bases (seed 0 vs
  three seeds), but the same sentence at L110 mixes 3-seed values at 16 / 20 / 24 with a seed-0 value at 32.
  *Rewrite:* use the 3-seed value everywhere, or append "(seed 0; −20 % over three seeds)".
- **m3. L32** — "and at best level with the best CFG point (12.24)" understates the paper's own row: `bucketu:12` at
  $w = 1.5$ is 10.45, better than best-CFG, and §4.6 builds its recommended alignment operating point on the same
  reference. *Rewrite:* "…10.45 / 12.10 at $w = 1.5 / 2$ — 1.9–3.5 FD worse than keeping the caption fixed, though
  still at or below the best CFG point".
- **m4. L468 (A.15 rule 3)** — the label's 16 px $w$-curve is called "flat" at worst/best ≈ 1.6× while the snapshot's
  1.59× at 24 px is called "steep". Report the ratios and drop the adjectives, or set one threshold.
- **m5. L427 (A.12)** — "The composed reference is not an artefact of the 100-step sampler" is shown downwards only:
  at 200 steps composed degrades 7.67 → 9.38 (+22 %) with no 200-step CFG or best-CFG comparison (only autoguidance
  $w=2$, 11.35 vs 11.33). *Rewrite:* "…is not an artefact of the step count in the 50–100 range; at 200 steps it
  degrades to 9.38, which we did not match with a 200-step CFG sweep."
- **m6. L48** — "No weight reaches FD < 12.2 at 16 px" generalises from an 8-point grid. The optimum is bracketed
  (16.39 / 12.24 / 12.98 at $w = 1 / 1.5 / 2$), so the claim is safe; one clause saying so ("the optimum is bracketed
  by $w = 1$ and $w = 2$") pre-empts the obvious reviewer request for $w = 1.25 / 1.75$.
- **m7. L139** — "[the CLIP cost] … is absent for reverse references". Table 4 gives reverse rows 29.58 / 29.60 at
  16 px against best-CFG 29.74: a 0.14–0.16 drop, of the same order as the composed row's 0.23 that the paragraph
  treats as real. Reverse references are also same-caption, so the stated cause ("guidance against a same-caption
  reference removes the unconditional CFG term") predicts they should pay the same cost. *Rewrite:* "…and is smaller
  for reverse references (−0.15 vs −0.23 at 16 px), which the missing-unconditional-term account does not fully
  explain."
- **m8. L30 / L40 / L216 — a discussion gap a reviewer will find.** Training feeds every sprite "BOX-downsampled to
  every lower bucket" (L40), so the lower-bucket label is, in training terms, the label of a box-downsampled view —
  and yet a trained 2×2 block-average branch (`probe_cg`) is the paper's most harmful reference, described at L30 as
  "the 16 px analogue of a low-frequency head". §5.4 gestures at the difference ("a block grid the strong prediction
  lacks") without connecting it to the data pipeline. One paragraph reconciling "box-downsampled targets under a
  lower label help" with "an explicit block-average branch is catastrophic" would close the most natural objection
  to the whole method.

---

## Sections that are sound

- **§3.1–§3.2 setup and rule.** The definition, the 2-NFE accounting and the "replaces CFG" claim are internally
  consistent and match Table A9.
- **§4.4 reverse controls (L112) and A.14.** The logic — if a wrong label acted as a generic unconditional-like
  reference, direction would not matter — is clean, the prediction is sharp, and it is confirmed at five resolutions
  and on a second model. With the `bucketmix` row it also rules out the ICG reading. This is the strongest piece of
  non-headline evidence in the paper and needs no repair.
- **§4.5's framing** ("replication within one family, not generality") and **§6's scope paragraph** are unusually
  honest and pre-empt most of the generalisation objections; **A.16** discloses the contaminated model rather than
  burying it.
- **§4.7's Inception paragraph** correctly names its own exceptions (v7s at 16 px, label-only flat at 24 / 32 px)
  rather than smoothing them — the earlier overstatement flagged as pass 2's c3 has been fixed.
- **A.11 cost** and **A.13's pre-registered failure criterion** ("Success criterion set in advance: FD < 7.2") are
  good practice and read as such.

---

## The four sentences a reviewer will attack first

1. **L7, "…at equal or better CLIP alignment than the CFG optimum at 16 px."** *Why:* Table 4 shows the composed row
   at 29.51 vs 29.74; the 29.77 belongs to a different row with a different FD. A reviewer who checks one number in
   the abstract will check this one, and finding it wrong colours the whole read. *Fix:* M1's rewrite.
2. **L218, "The belief's difference from the strong prediction is therefore *spatially structured* … not a global
   scalar."** *Why:* the control's failure mode (74.8 → 200.4 → 374.6 → 444.4) is the signature of a compounding
   fixed per-step gain, and the paper offers no gain-matched comparison. A diffusion-savvy reviewer will say the
   control was mis-specified and that the sharpest test in the paper therefore shows nothing. *Fix:* C3's rewrite —
   run the gain-matched version, or state the ambiguity.
3. **L15/L102, "lower contrast is exactly the direction of the error" / "the guided rows reduce the bare model's …
   insufficient local contrast."** *Why:* the paper's own Table 6 has the bare model *above* real in TV and the
   guided rows *below* the bare model. A reviewer reading §5 before §1 will see the framing invert. *Fix:* C2's
   rewrite.
4. **L30/L141, "PAG … tracks the CFG curve at every weight" / "PAG lies on the CFG curve, not the guided frontier."**
   *Why:* the direct zero-training competitor is dismissed by raw FD while the paper's own advantage is argued at
   matched CLIP; Table A2 shows PAG ~20 % below the CFG curve at its own alignment. A reviewer who owns PAG will
   spot this immediately. *Fix:* M2's rewrite, which still leaves the paper comfortably ahead.

---

## Overall judgement

**The empirical argument holds; the explanatory argument does not.**

The claim chain that matters — a bucketed model's lower-bucket label yields a reference that, used in place of CFG's
unconditional branch, beats re-tuned CFG at 16–32 px on two models, two feature spaces and a quantised variant; that
the effect is directional (higher buckets hurt); and that composing the label with an early snapshot is better still
— is supported, adequately controlled, and honestly bounded by §6. Items C4, C5, M1, M3, M4, M5 and most of the
Moderate list are over-statements of a real result and are fixable by rewording alone, without touching a number.

What does not hold is §5, the layer that turns the result into an explanation. Its central statistic is compared
across mismatched sampling regimes (C1), the failure mode it claims to correct is the opposite of what the same
table shows (C2), its one intervention admits a mundane alternative reading that was never excluded (C3), and its
"operational rule" (M7, M8) has been demonstrated only where the answer was already known. The paper's own hedges
("a characterisation, not a mechanism", L200, L234) cover the epistemic status but not the arithmetic of the
comparison: a reviewer who redoes the TV comparison against the regime-matched baseline gets a different — and in
one case sign-reversed — picture.

**Weakest link: the contrast/TV account in §5, and specifically the choice of CFG $w = 4$ samples as
$\mathrm{TV_{strong}}$.** Everything decorative in the paper hangs from it — the title's "bad model" reading, the
abstract's closing sentence, the intro's diagnosis, Figure 1's caption, Figures 3 and A6, and A.15's selection rule
— and it is the one part that a single cheap run could either repair or retire: sample the correct-label belief at
`--cfg 0` at 16 / 20 / 24 px, and re-run `shrink:f` at a matched effective gain. If the regime-matched
$\mathrm{TV_{strong}}$ is 26.2, the snapshot sits on the wrong side of the paper's own threshold, and the honest
version of §5 is "lower TV among same-caption beliefs correlates with usefulness; the snapshot shows TV is not the
whole story" — weaker, but unattackable, and it costs the paper nothing that §4 has not already earned.
