#!/usr/bin/env bash
# Stage 8 (after stage 7): expressibility of the reward through the cross-feature head inputs (CPU-heavy, run alone).
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until grep -q "cross_distill_played_b0.3" ../logs/diag_cross.log 2>/dev/null || grep -q Traceback ../logs/diag_cross.log 2>/dev/null; do sleep 60; done
$PY bottleneck.py --cross-only > ../logs/bottleneck_cross.log 2>&1
