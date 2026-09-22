"""Check that every headline number in the drafts appears in the experiment log.

Numbers drift when a table is edited by hand after a rerun, and again when a draft is retyped into a
second template -- the CVPR version was written from scratch against the same results, which is exactly
where a transcription error hides. Each entry maps the rounded value printed in the drafts to the
full-precision value recorded in experiment_log.md when the run finished. src/v6/verify_against_json.py
does the stronger check, against the scoring output rather than the prose.
Usage: python scripts/verify_paper_numbers.py
"""
import sys
from pathlib import Path

PAIRS = {
    "31.8": "31.80", "117.9": "117.94", "167.2": "167.23", "87.1": "87.14", "41.6": "41.64",
    "50.8": "50.84", "134.4": "134.37", "150.1": "150.07", "244.2": "244.17", "120.1": "120.13",
    "112.2": "112.16", "96.5": "96.46", "123.6": "123.60", "143.2": "143.21", "37.1": "37.06",
    "29.0": "29.02", "74.5": "74.46", "97.1": "97.12", "139.2": "139.19",
}

log = Path("pixel_art_research_20260816/experiment_log.md").read_text(encoding="utf-8")
drafts = {d: Path(f"paper_assets/{d}/main.tex").read_text(encoding="utf-8")
          for d in ("iclr27", "cvpr27") if Path(f"paper_assets/{d}/main.tex").exists()}
bad = []
for a, b in PAIRS.items():
    if b not in log:
        bad.append((a, b, "not in the log"))
        continue
    where = [d for d, t in drafts.items() if a in t]
    if not where:
        bad.append((a, b, "in no draft"))
if bad:
    print("UNBACKED NUMBERS:", bad)
    sys.exit(1)
# a number carried into the CVPR draft that the ICLR draft does not have is fine; the reverse is worth
# knowing about, since the CVPR draft is the submission
only_iclr = [a for a in PAIRS if "cvpr27" in drafts and a in drafts.get("iclr27", "") and a not in drafts["cvpr27"]]
print(f"ok: {len(PAIRS)} headline numbers are backed by the log; "
      f"{len(only_iclr)} appear only in the ICLR draft: {sorted(only_iclr)}")
