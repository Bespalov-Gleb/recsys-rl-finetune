#!/usr/bin/env bash
# Stage 5: seed replicates of the key GRPO runs (run-to-run variance of RL training).
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
for s in 1 2; do
  for b in 0.3 0.1; do
    $PY rl.py --reward played --beta $b --seed $s --tag grpo_played_b${b}_s$s > ../logs/grpo_played_b${b}_s$s.log 2>&1
  done
done
