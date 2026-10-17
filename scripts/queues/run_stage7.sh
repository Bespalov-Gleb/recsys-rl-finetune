#!/usr/bin/env bash
# Stage 7 (after stage 6): own idea that follows from the bottleneck finding -- give the policy the "user x track"
# features. Frozen GPT-2 + cross-feature head over the reference top-1000; RL (GRPO) at several beta and a supervised
# distillation of pi* as the ceiling of this architecture.
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until grep -q "distill_played_b0.3_lr1e-4" ../logs/strict_policies.log 2>/dev/null || grep -q Traceback ../logs/strict_policies.log ../logs/strict_sweep.log 2>/dev/null; do sleep 60; done
run() { tag=$1; shift; $PY "$@" --tag $tag > ../logs/$tag.log 2>&1; }
for b in 0.3 0.1 1.0 0.0; do
  run cross_played_b$b rl.py --policy cross --reward played --beta $b --lr 1e-3 --steps 3000 --eval_every 500
done
run cross_distill_played_b0.3 rl.py --policy cross --reward played --beta 0.3 --pg distill --lr 1e-3 --steps 3000 --eval_every 500
run cross_seconds_b0.3 rl.py --policy cross --reward seconds --beta 0.3 --lr 1e-3 --steps 3000 --eval_every 500
$PY diag_global.py cross_played_b0.3 cross_played_b0.1 cross_played_b1.0 cross_distill_played_b0.3 > ../logs/diag_cross.log 2>&1
