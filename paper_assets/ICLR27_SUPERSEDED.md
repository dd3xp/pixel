# `iclr27/` is superseded by `cvpr27/`

The target moved to CVPR 2027 on 2026-09-22. `paper_assets/cvpr27/` is the draft being prepared;
`paper_assets/iclr27/` is kept because it holds the longer treatment of several results and because the
ICLR abstract was registered from it.

**Do not submit or circulate `iclr27/main.pdf` without applying the corrections below.** The self-review
on 2026-09-22 found eight statements in the CVPR draft that needed fixing, all of which favoured us.
Each was then checked against this draft rather than assumed; the table says what is actually here.

| Issue | State in `iclr27/` |
|---|---|
| "44\% win" for cross-resolution guidance, a v7h-era number (6.24 vs 11.10) quoted beside v8n tables | present, lines 56 and 123 |
| Claims about what "this area" does, which we cannot cite | present, 3 places |
| The palette-size sentence does not match the measured 8/4/8/6 under either reading of "coarser" | present, line 591 |
| "2.6--4.8$\times$ further" where the low end is 2.49 | present, line 632 |
| "A larger step than the entire distance between the guidance rules" (55.3 vs 83.2) | present, line 421 |
| Palette figure caption mixes two guidance weights (87.2 with 36.9 and 34.1) | present |
| Prose 49 colours vs table 34 without stating the two populations | **not present** --- this draft's caption already scopes the 49 to the artwork feeding the bucket |
| "The biased column moves in the opposite direction throughout" | **not present** --- this draft does not make the claim |

Already corrected in both drafts, because they were number errors rather than framing: the leakage rate
(15.5\% → 16.2\%), the flat-neighbour definition, the unpaired real-sprite CLIP baseline, the
20/24\,px alignment cost, the palette-size "sanity check", and the robustness bound (less than 3 → at
most 3.0).

The registered ICLR abstract predates two corrections (167.2 → 117.9, and the 44\% figure); if that
submission is ever revived, the abstract has to be edited before the full-paper deadline.
