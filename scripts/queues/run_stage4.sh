#!/usr/bin/env bash
# Stage 4 (after stage 3): can the policy network represent the analytic optimum at all? Distil pi* into it.
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until [ -f ../results/eval2_exact_paper2_b0.0.json ] || grep -q Traceback ../logs/sweep2.log 2>/dev/null; do sleep 60; done
run() { tag=$1; shift; $PY "$@" --tag $tag > ../logs/$tag.log 2>&1; }
run distill_played_b0.3 rl.py --reward played --beta 0.3 --pg distill --lr 5e-4 --steps 3000 --eval_every 500
run distill_played_b0.1 rl.py --reward played --beta 0.1 --pg distill --lr 5e-4 --steps 3000 --eval_every 500
$PY diag_global.py distill_played_b0.3 distill_played_b0.1 > ../logs/diag_distill.log 2>&1
