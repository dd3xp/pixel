# Supplementary code: what reproduces which number

Every number in the paper comes from one of these entry points. Paths are relative to the repository root.
Runs assume the sprite corpus under `data/` and write to `runs_out/`.

The small artefacts a reader needs in order to check our claims without rebuilding the corpus are shipped in
`release_data/`: the per-file licence manifests, the duplicate and keep lists, the hold-out exclusion list, and the
corpus report. Everything else is rebuilt from public archives by the scripts below.

## 1. Corpus

| Step | Script | Output |
|---|---|---|
| Cut sheets into sprites, with provenance | `src/v6/extract_oga_v3.py <src> <out> <licence>` | `data/oga3_cut_*/` plus a `provenance.csv` per cut directory |
| Clean, cross-corpus pixel-hash dedup, report, contact sheet | `src/v6/build_corpus_v8.py` | `data/oga3_clean/`, `release_data/oga3_manifest.csv`, `release_data/corpus_v8_dupdrop.txt`, `release_data/corpus_v8_report.json` |
| Verify any further bundle against the same checklist | `src/v6/build_corpus_v9.py --new <dir>:<licence> --out <dir> --tag <name>` | `runs_out/corpus_<tag>_report.json`, `_sheet.png` |
| Drop font/UI sheets, collapse animation frames, cap per sheet | `src/v6/filter_corpus_v8.py --cap 60` | `release_data/oga3_keep.txt` |
| Caption new sprites (vision model, numbered grids) | `baseline/recaption.py --imgs <dir>` | `runs/recaptions_*.jsonl` |
| Label sprite content | `baseline/classify_sprites.py` (images), `baseline/classify_captions.py` (captions, 40 per batch) | `runs/*cls*.jsonl` |
| Rate craft 1--5 for the quality ablation | `baseline/score_quality.py` | `runs/quality_*.jsonl` |
| Turn craft ratings into caption tags | `src/v6/tag_quality.py` | `data/*_q.csv` |
| How native is a corpus's 16 px bucket | `src/v6/bucket_mix.py` | printed |

## 2. Training

| Model | Command |
|---|---|
| Main model **v8n** (rebuilt targets + new sprites, 80k) | `OUT=workdir/v8n DROP=release_data/corpus_v8_dupdrop.txt STEPS=80000 EXTRA_SRC=data/oga3_clean,data/oga3_captions_recap.csv,1 bash baseline/run_v8full.sh` |
| Control **v8f2** (no new sprites) | same without `EXTRA_SRC` |
| Control **v7r** (original targets) | `bash baseline/run_v7r.sh` |
| Enlarged corpus (the 20k/80k reversal) | `baseline/run_v8m.sh` (20k), `baseline/run_v8p.sh` (80k) |
| Transformer backbone | `baseline/run_v8dit.sh`, i.e. `--arch dit` through `train_v7.py` |
| Other ablations | `--width 192` (capacity), `CSV_SUFFIX=_q` (quality tags), `DROP=data/drop_dup_content.txt` (content filter), `--text_model openai/clip-vit-large-patch14` |

`src/v6/train_v8.py` wraps `src/v6/train_v7.py` (which also carries `--arch`) and adds `--max_down`, `--quant`, `--dupdrop`.

## 3. Sampling

`src/v6/sample_e.py`. The flags used in the paper:

- `--guide_mode icg`, `--cfg` — independent-condition guidance and its weight.
- `--pal K --pal_from f` — palette-projected sampling: project $\hat x_0$ onto a per-sprite $K$-colour palette for
  every step with $t/T \le f$. `--pal_mode global|median` are the two controls, `--pal_dist` draws $K$ per sprite.
- `--init_dir --init_strength` — apply the rule to another system's sprites instead of sampling from noise.
- `--guide_mode bucket:12`, `--guide_ckpt` — the cross-resolution references of the negative result.

## 4. Evaluation

| Protocol | Script |
|---|---|
| Biased reference (corpus sample) | `src/v6/fd_fair.py --size R --gen <dir>` |
| **Native reference** | `src/v6/fd_fair.py --size R --native [--pool union]` |
| Object-only reference | `src/v6/fd_ref.py --size R --ref runs_out/ref_object_s16 --gen <dir>` |
| Inception clean-FID / KID, either reference | `src/v6/fid_kid_fair.py --size R [--native]` |
| Text alignment (distractors re-seeded per set) | `src/v6/clip_score.py --size R --dirs name=<dir> ...` |
| Palette and flatness statistics | `src/v6/stats_simplicity.py --dirs name=<dir> ...`, `scripts/flat_check.py` |
| Retention of a refined sprite | `src/v6/refine_fidelity.py --init <dir> --runs <dirs>` |
| End-to-end matched run | `baseline/eval_matched_r.sh <tag> <R> <gpu>` |
| Palette size on a validation split | `baseline/k_split.sh`, `baseline/f_sweep.sh` |

## 5. External baselines

`baseline/api_gen.py` (gpt-image-2), `baseline/flux2_lora_gen.py` (FLUX.2-klein + pixel-art LoRA),
`baseline/sdxl_lora_gen.py` (SDXL + Pixel Art XL), `baseline/mys_sprites.py` (learned pixelization),
`baseline/pixeloe_sprites.py` (PixelOE), `baseline/run_sdpixl_recap.sh` (SD-piXL). All are converted to sprites by
`baseline/api_to_sprites.py` and given the same palette post-process by `src/v6/quant_dirs.py`.
`baseline/refine_ext.sh`, `baseline/refine_sweep.sh` and `baseline/refine_r.sh` run the decoding rule on their output.

## 6. Checking the paper against the runs

| Check | Script |
|---|---|
| Headline numbers against the scoring output | `src/v6/verify_against_json.py` |
| Headline numbers present in both drafts and the log | `scripts/verify_paper_numbers.py` |
| Corpus statistics recomputed from the data | `src/v6/audit_corpus_claims.py` |
| Duplicate and leakage figures | `src/v6/audit_leakage.py`, `scripts/leak_variants.py` |
| Anonymised snapshot for double-blind review | `scripts/make_anon_release.py` |

## Notes

- The evaluation hold-out is `release_data/holdout_exclude.txt`; any prompt set must be a subset of it
  (`src/v6/native_prompts.py` enforces this, after a bug where it did not).
- Licences: the sprites added in this work carry per-file records (`release_data/oga3_manifest.csv`,
  `release_data/kenney_manifest.csv`, `release_data/oga5_clean_manifest.csv`); the older material predates that
  bookkeeping.
- `release_data/corpus_v8_report.json` records the duplicate count the paper quotes: 43{,}342 paths collapse to
  38{,}454 unique sprites, i.e. 4{,}888 extra copies.
