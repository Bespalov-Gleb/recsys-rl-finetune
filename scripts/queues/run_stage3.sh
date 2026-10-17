#!/usr/bin/env bash
# Stage 3 (runs after grpo_played_b0.3_long and seq5_full_real, which were started by stage 2d):
# remaining multi-step GRPO, GRPO diagnostics (more samples, exact gradient, frozen embeddings, support
# constraint), softer DPO, drift analysis of RL policies, test-set sweep of the analytic optimum.
set -x
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
for t in grpo_played_b0.3_long seq5_full_real; do
  until [ -f ../results/eval2_$t.json ] || grep -q Traceback ../logs/$t.log; do sleep 60; done
done
run() { tag=$1; shift; $PY "$@" --tag $tag > ../logs/$tag.log 2>&1; }
run seq5_full_imag rl.py --reward full --beta 0.3 --T 5 --B 16 --G 8 --rm_ctx imagined --steps 1000 --eval_every 200
run seq5_full_imag_div rl.py --reward full --beta 0.3 --T 5 --B 16 --G 8 --rm_ctx imagined --div 0.2 --steps 1000 --eval_every 200
run grpo_seconds_b0.3_top1000 rl.py --reward seconds --beta 0.3 --topk 1000
run grpo_played_b0.1_top1000 rl.py --reward played --beta 0.1 --topk 1000
run grpo_played_b0.0_top1000 rl.py --reward played --beta 0.0 --topk 1000
run epg_played_b0.3 rl.py --reward played --beta 0.3 --pg exact --lr 5e-4 --steps 3000 --eval_every 500
run grpo_played_b0.3_frozen rl.py --reward played --beta 0.3 --freeze_emb --lr 5e-4 --steps 3000 --eval_every 500
run grpo_played_b0.3_long_G64 rl.py --reward played --beta 0.3 --lr 5e-4 --steps 3000 --G 64 --eval_every 500
run dpo_b1 baselines.py dpo --steps 1500 --lr 1e-4 --beta 1.0
$PY multistep.py base grpo_full_b0.3 grpo_played_b0.1 grpo_played_b0.0 grpo_played_b0.3_long seq5_full_real seq5_full_imag seq5_full_imag_div posce > ../logs/multistep_rl.log 2>&1
$PY rerank.py cache 2 > ../logs/cache2.log 2>&1
$PY rerank.py testsweep 2 > ../logs/sweep2.log 2>&1
