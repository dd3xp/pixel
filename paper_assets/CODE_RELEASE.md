# Supplementary code: what reproduces which number

Every number in the paper comes from one of these entry points. Paths are relative to the repository root; all runs
assume the sprite corpus under `data/` and write to `runs_out/`.

## 1. Corpus

| Step | Script | Output |
|---|---|---|
| Cut OpenGameArt sheets into sprites (any licence bundle, with provenance) | `src/v6/extract_oga_v3.py <src> <out> <licence>` | `data/oga3_cut_*/`, `provenance.csv` |
| Clean, cross-corpus pixel-hash dedup, verification report and contact sheet | `src/v6/build_corpus_v8.py` | `data/oga3_clean/`, `data/oga3_manifest.csv`, `data/corpus_v8_dupdrop.txt`, `runs_out/corpus_v8_report.json` |
| Drop font/UI sheets, collapse animation frames, cap per source sheet | `src/v6/filter_corpus_v8.py --cap 60` | `data/oga3_keep.txt` |
| Caption the new sprites (vision model, numbered grids) | `baseline/recaption.py --imgs <dir> --model gemini-3.8-flash` | `runs/recaptions_new.jsonl` |
| Label sprite content (object / effect / fragment / glyph / junk) | `baseline/classify_sprites.py` (images) or `baseline/classify_captions.py` (captions) | `runs/cls_*.jsonl` |
| Rate sprite craft 1--5 for the quality-conditioning ablation | `baseline/score_quality.py` | `runs/quality_*.jsonl` |
| Turn craft ratings into caption tags | `src/v6/tag_quality.py` | `data/*_q.csv` |

## 2. Training

| Model | Command |
|---|---|
| Main model **v8n** (rebuilt targets + new sprites, 80k steps) | `OUT=workdir/v8n DROP=data/corpus_v8_dupdrop.txt STEPS=80000 EXTRA_SRC=data/oga3_clean,data/oga3_captions_recap.csv,1 bash baseline/run_v8full.sh` |
| Control **v8f2** (no new sprites) | same without `EXTRA_SRC` |
| Control **v7r** (original targets) | `bash baseline/run_v7r.sh` |
| Ablations | `--width 192` (capacity), `CSV_SUFFIX=_q` (quality tags), `DROP=data/drop_dup_content.txt` (content filter), `--text_model openai/clip-vit-large-patch14` (text encoder) |

`src/v6/train_v8.py` wraps `src/v6/train_v7.py` and adds `--max_down`, `--quant`, `--dupdrop`.

## 3. Sampling

`src/v6/sample_e.py` is the sampler. The flags used in the paper:

- `--guide_mode icg` --- independent-condition guidance; `--cfg` is the guidance weight.
- `--pal K --pal_from f` --- palette-projected sampling (Section 3.2): project $\hat x_0$ onto a per-sprite
  $K$-colour palette for every step with $t/T \le f$. `--pal_mode global|median` are the two controls,
  `--pal_dist` draws $K$ per sprite from the corpus histogram.
- `--guide_mode bucket:12` / `--guide_ckpt` --- the cross-resolution references of the negative result.

## 4. Evaluation

| Protocol | Script |
|---|---|
| Biased reference (corpus sample) | `src/v6/fd_fair.py --size R --gen <dir>` |
| **Native reference** (sprites natively $\le R$ px) | `src/v6/fd_fair.py --size R --native [--pool union]` |
| Object-only reference (content-filtered) | `src/v6/fd_ref.py --size R --ref runs_out/ref_object_s16 --gen <dir>` |
| Text alignment | `src/v6/clip_score.py --size R --dirs name=<dir> ...` |
| End-to-end matched run (sample + score) | `baseline/eval_matched_r.sh <tag> <R> <gpu>` |
| Native ground-truth prompt set (held-out, natively small) | `src/v6/native_prompts.py --size 16` |
| Paper numbers vs the experiment log | `scripts/verify_paper_numbers.py` |

## 5. External baselines

`baseline/api_gen.py` (gpt-image-2, nano-banana-2), `baseline/flux2_lora_gen.py` (FLUX.2-klein + pixel-art LoRA),
`baseline/sdxl_lora_gen.py` (SDXL + Pixel Art XL), `baseline/mys_sprites.py` (learned pixelization),
`baseline/pixeloe_sprites.py` (PixelOE). All are converted to sprites by `baseline/api_to_sprites.py` and given the
same palette post-process by `src/v6/quant_dirs.py`.

## Notes

- The evaluation hold-out is `runs_out/holdout_exclude.txt`; any prompt set must be a subset of it
  (`src/v6/native_prompts.py` enforces this, after a bug where it was not).
- Licences: the sprites added in this work carry per-file licence records
  (`data/oga3_manifest.csv`, `data/kenney_manifest.csv`); the older material predates that bookkeeping.
