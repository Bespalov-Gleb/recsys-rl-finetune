#!/usr/bin/env bash
# Stage 2b: RM-free baselines (Positive-CE, DPO) and the multi-step drift analysis of the base model.
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until grep -q "rerank.py sweep 1" ../logs/stage2a.log && ! pgrep -f "rerank.py sweep" >/dev/null 2>&1 && [ -f ../results/eval1_exact_paper2_b0.0.json ]; do sleep 30; done
$PY baselines.py posce --steps 3000 --lr 1e-4 > ../logs/posce.log 2>&1
$PY baselines.py dpo --steps 1500 --lr 1e-4 --beta 0.1 > ../logs/dpo.log 2>&1
$PY multistep.py base > ../logs/multistep_base.log 2>&1
