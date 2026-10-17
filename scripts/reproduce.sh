#!/usr/bin/env bash
# Every experiment of the report, in the order it was run (the run_stage*.sh files are the queues actually used;
# this file is the clean sequential version). One GPU job at a time: on an 8 GB card two trainings spill into
# shared memory and slow down ~10x. Total ~15 h on RTX 4060 Laptop.
set -ex
cd "$(dirname "$0")/../src"
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
RAW=${RAW:-../data/raw/yambda_50m}   # multi_event.parquet, artist_item_mapping.parquet from yandex/yambda (flat/50m)

# data + leakage tests
$PY prep.py $RAW ../data
$PY split_npz.py
$PY ../tests/test_leakage.py

# base policy, reward models, gold judge
$PY policy.py train --steps 30000 --tag base
$PY rm.py train --tag rm0 --steps 20000 && $PY rm.py eval --tag rm0
$PY rm.py train --tag rm1 --seed 1 --boot --steps 12000 && $PY rm.py eval --tag rm1
$PY rm.py train --tag rm2 --seed 2 --boot --steps 12000 && $PY rm.py eval --tag rm2
$PY gold.py train && $PY gold.py eval

# reference points, candidate cache, RM diagnostics, analytic optimum (val)
$PY simple.py
$PY rerank.py cache 1
$PY rm_analysis.py
$PY rerank.py sweep 1
$PY rerank.py full 1
$PY rerank.py pess 1

# RM-free baselines
$PY baselines.py posce --steps 3000 --lr 1e-4
$PY baselines.py dpo --steps 1500 --lr 1e-4 --beta 0.1
$PY baselines.py dpo --steps 1500 --lr 1e-4 --beta 1.0 --tag dpo_b1

# single-step GRPO
for b in 0.3 1.0 0.1 0.0; do $PY rl.py --reward played --beta $b --tag grpo_played_b$b; done
for r in full seconds played_db like lt paper2; do $PY rl.py --reward $r --beta 0.3 --tag grpo_${r}_b0.3; done
$PY rl.py --reward played --beta 0.1 --ens rm0,rm1,rm2 --pess 1.0 --tag grpo_played_b0.1_pess1
for s in 1 2; do for b in 0.3 0.1; do $PY rl.py --reward played --beta $b --seed $s --tag grpo_played_b${b}_s$s; done; done
$PY rl.py --reward played --beta 0.3 --lr 5e-4 --steps 6000 --eval_every 500 --tag grpo_played_b0.3_long
$PY rl.py --reward played --beta 0.3 --lr 5e-4 --steps 3000 --G 64 --eval_every 500 --tag grpo_played_b0.3_long_G64
$PY rl.py --reward played --beta 0.3 --pg exact --lr 5e-4 --steps 3000 --eval_every 500 --tag epg_played_b0.3
$PY rl.py --reward played --beta 0.3 --freeze_emb --lr 5e-4 --steps 3000 --eval_every 500 --tag grpo_played_b0.3_frozen
$PY rl.py --reward seconds --beta 0.3 --topk 1000 --tag grpo_seconds_b0.3_top1000
$PY rl.py --reward played --beta 0.1 --topk 1000 --tag grpo_played_b0.1_top1000
$PY rl.py --reward played --beta 0.0 --topk 1000 --tag grpo_played_b0.0_top1000
# can the network represent the analytic optimum at all? supervised distillation of pi*
$PY rl.py --reward played --beta 0.3 --pg distill --lr 5e-4 --steps 3000 --eval_every 500 --tag distill_played_b0.3
$PY rl.py --reward played --beta 0.1 --pg distill --lr 5e-4 --steps 3000 --eval_every 500 --tag distill_played_b0.1
$PY rl.py --reward played --beta 0.3 --pg distill --lr 1e-4 --steps 3000 --eval_every 500 --tag distill_played_b0.3_lr1e-4

# multi-step GRPO and drift analysis
$PY rl.py --reward full --beta 0.3 --T 5 --B 16 --G 8 --rm_ctx real --steps 1000 --eval_every 200 --tag seq5_full_real
$PY rl.py --reward full --beta 0.3 --T 5 --B 16 --G 8 --rm_ctx imagined --steps 1000 --eval_every 200 --tag seq5_full_imag
$PY rl.py --reward full --beta 0.3 --T 5 --B 16 --G 8 --rm_ctx imagined --div 0.2 --steps 1000 --eval_every 200 --tag seq5_full_imag_div
$PY multistep.py base grpo_full_b0.3 grpo_played_b0.1 grpo_played_b0.0 grpo_played_b0.3_long seq5_full_real seq5_full_imag seq5_full_imag_div posce

# diagnostics: alignment of the policy change with the reward (val and train contexts), softmax bottleneck
$PY diag_global.py grpo_played_b0.3 grpo_played_b0.1 grpo_played_b0.3_long grpo_played_b0.3_long_G64 epg_played_b0.3 grpo_played_b0.3_frozen posce dpo dpo_b1 distill_played_b0.3 distill_played_b0.1 distill_played_b0.3_lr1e-4
$PY diag_global.py --train grpo_played_b0.3 grpo_played_b0.1 grpo_played_b0.3_long grpo_played_b0.3_long_G64 epg_played_b0.3 grpo_played_b0.3_frozen
$PY bottleneck.py

# own idea: give the policy the "user x track" features (frozen GPT-2 + cross-feature head over the top-1000)
for b in 0.3 0.1 1.0 0.0; do $PY rl.py --policy cross --reward played --beta $b --lr 1e-3 --steps 3000 --eval_every 500 --tag cross_played_b$b; done
$PY rl.py --policy cross --reward played --beta 0.3 --pg distill --lr 1e-3 --steps 3000 --eval_every 500 --tag cross_distill_played_b0.3
$PY rl.py --policy cross --reward seconds --beta 0.3 --lr 1e-3 --steps 3000 --eval_every 500 --tag cross_seconds_b0.3
$PY diag_global.py cross_played_b0.3 cross_played_b0.1 cross_played_b1.0 cross_distill_played_b0.3
$PY bottleneck.py --cross-only

# test set: analytic optimum, bootstrap, figures
$PY rerank.py cache 2
$PY rerank.py testsweep 2
$PY bootstrap.py base exact_played_b1.0 exact_played_b0.3 exact_played_b0.1 exact_full_b0.3 exact_played_db_b0.3 exact_like_b1.0 exact_lt_b1.0 grpo_played_b1.0 grpo_played_b0.3 grpo_played_b0.1 grpo_full_b0.3 grpo_played_b0.3_long seq5_full_real posce dpo_b1 dpo histfreq
mv ../results/bootstrap_vs_base.json ../results/bootstrap_vs_base.main.tmp
$PY bootstrap.py base distill_played_b0.3_lr1e-4 distill_played_b0.3 epg_played_b0.3 grpo_played_b0.3_frozen
mv ../results/bootstrap_vs_base.json ../results/bootstrap_vs_base_diag.json && mv ../results/bootstrap_vs_base.main.tmp ../results/bootstrap_vs_base.json
# control: "repeat what the user likes" re-ranker; repeats vs discoveries
$PY heuristic.py val
for c in "2.0 1.0" "1.0 2.0" "2.0 4.0"; do $PY heuristic.py test $c; done
$PY novelty.py 2 base histfreq heur_a2.0_g1.0 heur_a1.0_g2.0 heur_a2.0_g4.0 exact_played_b1.0 exact_played_b0.3 exact_full_b0.3 exact_played_b0.1 cross_played_b1.0 cross_played_b0.3 cross_played_b0.1 cross_played_b0.0 cross_distill_played_b0.3 grpo_played_b0.3 grpo_played_b0.1 posce
$PY reco_targets.py base histfreq heur_a2.0_g1.0 heur_a1.0_g2.0 exact_played_b1.0 exact_played_b0.3 exact_played_b0.1 cross_played_b1.0 cross_played_b0.3 cross_played_b0.1 cross_distill_played_b0.3 grpo_played_b0.3 posce
$PY bootstrap.py --out=bootstrap_vs_base_cross base cross_played_b1.0 cross_played_b0.3 cross_played_b0.1 cross_played_b0.0 cross_distill_played_b0.3 heur_a2.0_g1.0 heur_a1.0_g2.0 exact_played_b1.0 exact_played_b0.3 exact_played_b0.1 histfreq

# robustness to the 5-second timestamp ties: histories strictly earlier in time
RS_STRICT=1 $PY rerank.py strict 2
RS_STRICT=1 $PY eval_policies.py _strict grpo_played_b1.0 grpo_played_b0.3 grpo_played_b0.1 posce dpo_b1 distill_played_b0.3_lr1e-4 cross_played_b0.3 cross_played_b0.1 cross_distill_played_b0.3
RS_STRICT=1 $PY bootstrap.py --out=bootstrap_vs_base_strict base_strict exact_played_b1.0_strict exact_played_b0.3_strict exact_full_b0.3_strict exact_played_b0.1_strict grpo_played_b1.0_strict grpo_played_b0.3_strict grpo_played_b0.1_strict posce_strict dpo_b1_strict distill_played_b0.3_lr1e-4_strict exact_like_b1.0_strict
RS_STRICT=1 $PY bootstrap.py --out=bootstrap_vs_base_strict_cross base_strict cross_played_b0.3_strict cross_played_b0.1_strict cross_distill_played_b0.3_strict

$PY figures.py
$PY build_report.py
