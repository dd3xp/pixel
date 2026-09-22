"""Materialise the held half of the native pool at 20 and 24 px.

The provenance probe needs it as a control: without it the probe reports an internal split accuracy and
nothing about unseen real sprites, which is exactly the check that disqualified the VLM judge.
"""
import sys
sys.path.insert(0, "src/v6")
from fd_fair import real_split, dump_totensor

for R in (20, 24):
    ref, held = real_split(native_R=R, pool="old")
    dump_totensor(held, f"runs_out/held_native{R}", R)
    print(f"{R}px: ref {len(ref)}, held {len(held)}", flush=True)
