#!/bin/bash
# Build the draft with tectonic (portable binary in the session scratchpad; any TeX Live pdflatex+bibtex also works:
#   pdflatex main && bibtex main && pdflatex main && pdflatex main).
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-tectonic}
"$TECTONIC" --keep-logs main.tex 2>&1 | grep -v "^note: downloading"
