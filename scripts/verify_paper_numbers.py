"""Check that every headline number in the paper appears in the experiment log.

Numbers drift when a table is edited by hand after a rerun; this is the cheap guard. Each entry maps the rounded
value printed in main.tex to the full-precision value recorded in experiment_log.md when the run finished.
Usage: python scripts/verify_paper_numbers.py
"""
import sys
from pathlib import Path

PAIRS = {
    "31.8": "31.80", "117.9": "117.94", "167.2": "167.23", "87.1": "87.14", "41.6": "41.64",
    "50.8": "50.84", "134.4": "134.37", "150.1": "150.07", "244.2": "244.17", "120.1": "120.13",
    "112.2": "112.16", "96.7": "96.67", "124.2": "124.20", "142.0": "142.04", "37.1": "37.06",
    "29.0": "29.02", "74.5": "74.46", "97.1": "97.12", "139.2": "139.19",
}

tex = Path("paper_assets/iclr27/main.tex").read_text(encoding="utf-8")
log = Path("pixel_art_research_20260816/experiment_log.md").read_text(encoding="utf-8")
bad = [(a, b) for a, b in PAIRS.items() if a not in tex or b not in log]
if bad:
    print("UNBACKED NUMBERS:", bad)
    sys.exit(1)
print(f"ok: {len(PAIRS)} headline numbers in the paper are backed by the log")
