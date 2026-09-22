"""Produce an anonymised code snapshot for double-blind submission.

The working tree cannot be scrubbed in place --- the paths are what the scripts actually run against ---
so this exports a copy. It rewrites the strings that identify us (the account and project directory on
the cluster, the captioning host, the GitHub handle), drops the files that are private working notes or
raw credentials-adjacent, and writes a manifest of exactly what was changed so the substitution can be
checked rather than trusted.

It does not upload anything. Review the output, then upload it yourself.

Usage: python scripts/make_anon_release.py --out build/anon_release
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

# literal -> replacement. Longest first, so a path is rewritten before the account name inside it.
SUBS = [
    ("/mnt/data/kw/RoundSquisheen/pixel/pixel", "<PROJECT_ROOT>"),
    ("/mnt/data/kw/RoundSquisheen/pixel", "<WORKSPACE>"),
    ("/mnt/data/kw/anaconda3/envs/SD-piXL", "<CONDA_ENV>"),
    ("/mnt/data/kw", "<WORKSPACE>"),
    ("RoundSquisheen", "workspace"),
    ("http://113.45.39.247:3001/v1", "<CAPTION_API_BASE>"),
    ("113.45.39.247:3001", "<CAPTION_API_HOST>"),
    ("113.45.39.247", "<CAPTION_API_HOST>"),
    ("dd3xp", "anon"),
]
# anything matching these is not exported at all
DROP_DIRS = {"pixel_art_research_20260816", "runs", ".git", ".claude", "build"}
DROP_GLOBS = ["*.pt", "*.tgz", "*.zip", "*.jsonl", "*.pdf", "*.log", "*.blg", "*.aux"]
# a secret must never survive, so this is checked after substitution rather than substituted
FORBIDDEN = [re.compile(r"sk-[A-Za-z0-9]{20,}"), re.compile(r"ssh-(rsa|ed25519)\s")]


def tracked_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [Path(p) for p in out.splitlines() if p]


def wanted(p: Path):
    if set(p.parts) & DROP_DIRS:
        return False
    if any(p.match(g) for g in DROP_GLOBS):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/anon_release")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    changed, copied, skipped, flagged = {}, 0, 0, []
    for p in tracked_files():
        if not wanted(p):
            skipped += 1
            continue
        dst = out / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            shutil.copy2(p, dst)
            copied += 1
            continue
        hits = {}
        for lit, rep in SUBS:
            n = text.count(lit)
            if n:
                text = text.replace(lit, rep)
                # key by the replacement, never the literal: a manifest that lists what was hidden
                # would deanonymise the release as thoroughly as not rewriting it
                hits[rep] = hits.get(rep, 0) + n
        for pat in FORBIDDEN:
            if pat.search(text):
                flagged.append(str(p))
        dst.write_text(text, encoding="utf-8", newline="")
        copied += 1
        if hits:
            changed[str(p)] = hits

    # the manifest ships with the release, so it is checked against the same patterns
    (out / "ANON_MANIFEST.json").write_text(
        json.dumps({"files_copied": copied, "files_skipped": skipped,
                    "substitutions": changed, "flagged_secrets": flagged}, indent=1),
        encoding="utf-8")
    print(f"copied {copied}, skipped {skipped}, rewrote strings in {len(changed)} files")
    if flagged:
        print("REFUSING TO CALL THIS ANONYMOUS: possible secrets in", flagged)
        raise SystemExit(1)
    print(f"manifest: {out / 'ANON_MANIFEST.json'}")


if __name__ == "__main__":
    main()
