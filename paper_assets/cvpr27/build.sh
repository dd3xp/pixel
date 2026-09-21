#!/bin/bash
# Build the CVPR draft. Same tectonic invocation as the ICLR draft; cvpr.sty is year-agnostic
# (the options are review/final/rebuttal), so the 2026 kit compiles a 2027 submission and only the
# .sty needs swapping when the 2027 kit appears.
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-tectonic}
"$TECTONIC" --keep-logs main.tex 2>&1 | grep -v "^note: downloading"
