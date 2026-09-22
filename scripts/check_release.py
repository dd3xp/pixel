"""Check that every file CODE_RELEASE.md names is actually in the anonymised release.

A release document that points at files the release does not contain is worse than no document: a
reviewer who tries to follow it concludes the work is not reproducible. Two names in the document are
prose rather than paths (`provenance.csv`, written per cut directory, and a bare `train_v7.py`), so they
are exempted explicitly rather than silently.

Usage: python scripts/make_anon_release.py && python scripts/check_release.py
"""
import re
import sys
from pathlib import Path

EXEMPT = {"provenance.csv", "train_v7.py"}
RELEASE = Path("build/anon_release")


def main():
    if not RELEASE.exists():
        sys.exit(f"{RELEASE} does not exist; run scripts/make_anon_release.py first")
    doc = Path("paper_assets/CODE_RELEASE.md").read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|sh|csv|txt|json))", doc)) - EXEMPT)
    missing = [r for r in refs if not (RELEASE / r).exists()]
    print(f"{len(refs)} files referenced by CODE_RELEASE.md, {len(EXEMPT)} exempt as prose")
    if missing:
        print("MISSING FROM THE RELEASE:", missing)
        sys.exit(1)
    print("ok: every referenced file is in the release")


if __name__ == "__main__":
    main()
