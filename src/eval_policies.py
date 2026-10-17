"""Re-evaluate saved policies on the test set (e.g. under RS_STRICT=1) and save as eval2_<tag><suffix>.

python src/eval_policies.py _strict grpo_played_b0.3 posce ...
"""
import sys

from evaluate import eval_set, full_eval, policy_topk
from policy import load_policy
from rm import load_rm

if __name__ == "__main__":
    suffix, tags = sys.argv[1], sys.argv[2:]
    ref, rm = load_policy("base").eval(), load_rm("rm0")
    idx = eval_set(2, 3)[0]
    for tag in tags:
        recs, d = policy_topk(load_policy(tag), idx, K=100, ref=ref)
        r = full_eval(tag + suffix, recs, split=2, per_user=3, extra=d, rms=[rm])
        print(tag, {k: round(r[k], 4) for k in ("HR@10", "Recall@10_full", "Recall@10_skip", "gold_played", "kl_ref")}, flush=True)
