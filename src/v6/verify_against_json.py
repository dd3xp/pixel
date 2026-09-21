"""Check the paper's headline numbers against the JSON the scoring scripts wrote, not against prose.

verify_paper_numbers.py compares main.tex with experiment_log.md, which catches a table edited after a
rerun but not a number that was wrong in both -- and on 09-21 two claims turned out to be exactly that,
written from memory into the log and the paper together.  This closes the loop: each entry names the
run directory and the reference protocol, and the value is read from runs_out/*.json on the server.

Run on the server (the JSON lives next to the samples):
  python src/v6/verify_against_json.py
"""
import json
import sys
from pathlib import Path

# paper value -> (json file, key substring[, "mean" if the paper reports a seed average])
# Three of the headline numbers are means over three seeds, so a single-seed lookup would "fail" them.
CHECKS = {
    "31.8":  ("runs_out/fair_fd16_native.json", "v8n_icg2_pal4_matched_eval", "mean"),
    "117.9": ("runs_out/fair_fd16_native.json", "v8n_cfg1p5_matched_eval"),
    "87.1":  ("runs_out/fair_fd16_native.json", "v8n_icg_w2_matched_eval"),
    "120.1": ("runs_out/fair_fd16_native.json", "v8n_icg2_globalpal4_matched_eval"),
    "112.2": ("runs_out/fair_fd16_native.json", "v8n_icg2_median_matched_eval"),
    "36.9":  ("runs_out/fair_fd16_native.json", "v8n_icg2_pal6_matched_eval"),
    "38.1":  ("runs_out/fair_fd16_native.json", "v8n_icg2_pal3_matched_eval"),
    "33.7":  ("runs_out/fair_fd16_native.json", "v8n_icg2_pal5_matched_eval"),
    "39.9":  ("runs_out/fair_fd16_native.json", "v8n_icg2_paldist_matched_eval"),
    "96.7":  ("runs_out/fair_fd20_native.json", "v8n_r20_icg2_pal8_matched_eval"),
    "124.2": ("runs_out/fair_fd24_native.json", "v8n_r24_icg2_pal6_matched_eval"),
    "143.2": ("runs_out/fair_fd12_native.json", "v8n_r12_icg2_pal8_matched_eval", "mean"),
    "39.3":  ("runs_out/fair_fd16_native.json", "v8p_icg2_pal4_matched_eval", "mean"),
    "42.9":  ("runs_out/fair_fd16_native.json", "v8m_20k_icg2_pal4_matched_eval"),
    "53.2":  ("runs_out/fair_fd16_native.json", "v8k_20k_icg2_pal4_matched_eval"),
    "54.2":  ("runs_out/fair_fd16_native.json", "v8n_20k_icg2_pal4_matched_eval"),
    "86.5":  ("runs_out/fair_fd16_native.json", "refine_gpt_s0p6"),
    "88.2":  ("runs_out/fair_fd16_native.json", "refine_flux2_s0p6"),
    "155.0": ("runs_out/fair_fd16_native.json", "refine_lora_s0p6"),
    "53.6":  ("runs_out/fair_fd16_native.json", "refine_gpt_s1p0"),
    # the external table and the no-post-process row, from the ext200 scoring files
    "50.8":  ("runs_out/ext200/fd_native_v8n.json", "icg2_pal4"),
    "134.4": ("runs_out/ext200/fd_native_q.json", "gpt_q4"),
    "150.1": ("runs_out/ext200/fd_native_q.json", "flux2_q4"),
    "244.2": ("runs_out/ext200/fd_native_q.json", "lora_q4"),
    "77.0":  ("runs_out/ext200/fd_native_q.json", "real_q4"),
    "284.4": ("runs_out/ext200/fd_native_s16.json", "ext_recap_gpt"),
    "318.9": ("runs_out/ext200/fd_native_s16.json", "ext_flux2"),
    "436.4": ("runs_out/ext200/fd_native_s16.json", "ext_lora"),
    "4.47":  ("runs_out/ext200/fd_native_s16.json", "floor_heldout3000"),
}
TOL = 0.06   # the paper rounds to one decimal


def main():
    cache, bad, ok = {}, [], 0
    for want, spec in CHECKS.items():
        f, key = spec[0], spec[1]
        if f not in cache:
            p = Path(f)
            cache[f] = json.loads(p.read_text()) if p.exists() else None
        d = cache[f]
        if d is None:
            bad.append((want, f, "json missing"))
            continue
        base = key.replace("_matched_eval", "")
        if len(CHECKS[want]) > 2 and CHECKS[want][2] == "mean":
            hits = [v for k, v in d.items() if base in k and ("seed" in k or base + "_matched_eval" in k)]
            if not hits:
                bad.append((want, key, "no such run"))
                continue
            got = sum(hits) / len(hits)
        else:
            hits = [v for k, v in d.items() if key in k]
            if not hits:
                bad.append((want, key, "no such run"))
                continue
            got = hits[0]
        if abs(got - float(want)) > TOL:
            bad.append((want, key, f"json says {got}"))
        else:
            ok += 1
    for b in bad:
        print("MISMATCH", b, flush=True)
    print(f"{ok}/{len(CHECKS)} headline numbers match the scoring output", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
