#!/usr/bin/env bash
# Stage 2a: reference points, candidate cache, RM diagnostics, analytic KL-optimum sweep on validation.
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until [ -f ../logs/rm2_eval.log ] && grep -q "full_gauc" ../logs/rm2_eval.log; do sleep 30; done
$PY simple.py > ../logs/simple.log 2>&1
$PY rerank.py cache 1 > ../logs/cache1.log 2>&1
$PY rm_analysis.py > ../logs/rm_analysis.log 2>&1
$PY rerank.py sweep 1 > ../logs/sweep1.log 2>&1
