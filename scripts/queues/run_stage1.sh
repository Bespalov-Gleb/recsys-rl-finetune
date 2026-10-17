#!/usr/bin/env bash
# Stage 1: base policy (already started separately) -> proxy reward model -> gold judge -> ensemble members.
# GPU has 8 GB: heavy trainings strictly one at a time (two at once spill into shared memory and crawl).
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until [ -f ../checkpoints/policy_base.pt ]; do
  sleep 60; done
$PY rm.py train --tag rm0 --steps 20000 > ../logs/rm0.log 2>&1
$PY rm.py eval --tag rm0 > ../logs/rm0_eval.log 2>&1
$PY gold.py train > ../logs/gold.log 2>&1 &
$PY rm.py train --tag rm1 --seed 1 --boot --steps 12000 > ../logs/rm1.log 2>&1
$PY rm.py train --tag rm2 --seed 2 --boot --steps 12000 > ../logs/rm2.log 2>&1
wait
$PY gold.py eval > ../logs/gold_eval.log 2>&1
for t in rm1 rm2; do $PY rm.py eval --tag $t > ../logs/${t}_eval.log 2>&1; done
