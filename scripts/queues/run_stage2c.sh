#!/usr/bin/env bash
# Stage 2c: single-step GRPO -- beta sweep on the played-ratio reward, then every reward at beta = 0.3
# (the knee of the analytic sweep), then the pessimistic ensemble reward.
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until [ -f ../results/multistep_base.json ] || grep -q Traceback ../logs/multistep_base.log 2>/dev/null; do sleep 30; done
for b in 0.3 1.0 0.1 0.0; do
  $PY rl.py --reward played --beta $b --tag grpo_played_b$b > ../logs/grpo_played_b$b.log 2>&1
done
for r in full seconds played_db like lt paper2; do
  $PY rl.py --reward $r --beta 0.3 --tag grpo_${r}_b0.3 > ../logs/grpo_${r}_b0.3.log 2>&1
done
$PY rl.py --reward played --beta 0.1 --ens rm0,rm1,rm2 --pess 1.0 --tag grpo_played_b0.1_pess1 > ../logs/grpo_played_b0.1_pess1.log 2>&1
