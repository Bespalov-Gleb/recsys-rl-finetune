#!/usr/bin/env bash
# Stage 2d: longer GRPO (does the policy close the gap to the analytic optimum?), DPO with a softer beta,
# unrestricted-candidate optimum, pessimistic optimum, multi-step GRPO, drift analysis of RL policies,
# test-set sweep of the analytic optimum.
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
until [ -f ../results/eval2_grpo_played_b0.1_pess1.json ] || grep -q Traceback ../logs/grpo_played_b0.1_pess1.log 2>/dev/null; do sleep 60; done
$PY rl.py --reward played --beta 0.3 --lr 5e-4 --steps 6000 --eval_every 500 --tag grpo_played_b0.3_long > ../logs/grpo_played_b0.3_long.log 2>&1
$PY rl.py --reward played --beta 0.3 --lr 5e-4 --steps 6000 --G 64 --eval_every 500 --tag grpo_played_b0.3_long_G64 > ../logs/grpo_played_b0.3_long_G64.log 2>&1
$PY baselines.py dpo --steps 1500 --lr 1e-4 --beta 1.0 --tag dpo_b1 > ../logs/dpo_b1.log 2>&1
$PY rerank.py full 1 > ../logs/full1.log 2>&1
$PY rerank.py pess 1 > ../logs/pess1.log 2>&1
$PY rl.py --reward full --beta 0.3 --T 5 --B 16 --G 8 --rm_ctx real --steps 1000 --eval_every 200 --tag seq5_full_real > ../logs/seq5_full_real.log 2>&1
$PY rl.py --reward full --beta 0.3 --T 5 --B 16 --G 8 --rm_ctx imagined --steps 1000 --eval_every 200 --tag seq5_full_imag > ../logs/seq5_full_imag.log 2>&1
$PY rl.py --reward full --beta 0.3 --T 5 --B 16 --G 8 --rm_ctx imagined --div 0.2 --steps 1000 --eval_every 200 --tag seq5_full_imag_div > ../logs/seq5_full_imag_div.log 2>&1
$PY multistep.py base grpo_full_b0.3 grpo_played_b0.3_long grpo_played_b0.0 seq5_full_real seq5_full_imag seq5_full_imag_div posce > ../logs/multistep_rl.log 2>&1
$PY rerank.py cache 2 > ../logs/cache2.log 2>&1
$PY rerank.py testsweep 2 > ../logs/sweep2.log 2>&1
# support constraint: the policy may only re-rank the reference top-1000 (as the analytic optimum does)
$PY rl.py --reward seconds --beta 0.3 --topk 1000 --tag grpo_seconds_b0.3_top1000 > ../logs/grpo_seconds_b0.3_top1000.log 2>&1
$PY rl.py --reward played --beta 0.1 --topk 1000 --tag grpo_played_b0.1_top1000 > ../logs/grpo_played_b0.1_top1000.log 2>&1
$PY rl.py --reward played --beta 0.0 --topk 1000 --tag grpo_played_b0.0_top1000 > ../logs/grpo_played_b0.0_top1000.log 2>&1
# why is sampled GRPO less KL-efficient than the analytic optimum? sampling noise vs representation
$PY rl.py --reward played --beta 0.3 --pg exact --lr 5e-4 --steps 3000 --eval_every 500 --tag epg_played_b0.3 > ../logs/epg_played_b0.3.log 2>&1
$PY rl.py --reward played --beta 0.3 --freeze_emb --lr 5e-4 --steps 3000 --eval_every 500 --tag grpo_played_b0.3_frozen > ../logs/grpo_played_b0.3_frozen.log 2>&1
