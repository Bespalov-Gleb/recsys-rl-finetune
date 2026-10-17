#!/usr/bin/env bash
# Stage 6 (after the seed replicates): robustness of the test conclusions to the 5-second timestamp ties --
# every history ends strictly before the decision time (RS_STRICT=1).
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until [ -f ../results/eval2_grpo_played_b0.1_s2.json ] || grep -q Traceback ../logs/grpo_played_b0.1_s2.log 2>/dev/null; do sleep 60; done
export RS_STRICT=1
$PY rerank.py strict 2 > ../logs/strict_sweep.log 2>&1
$PY eval_policies.py _strict grpo_played_b1.0 grpo_played_b0.3 grpo_played_b0.1 posce dpo_b1 distill_played_b0.3_lr1e-4 > ../logs/strict_policies.log 2>&1
