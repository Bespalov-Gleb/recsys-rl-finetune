"""How personalised is the change that fine-tuning makes?

For a fine-tuned policy, delta(s, a) = log pi(a|s) - log pi_ref(a|s) on the reference top-1000 candidates of
2000 val contexts. We decompose delta into a per-item constant (the same boost for every user) plus a per-context
shift and a residual; R^2 of "context shift + item constant" tells how much of the change is a global item re-weighting
rather than context-specific re-ranking. The same is done for the analytic optimum, where delta = r(s,a)/beta + const(s).

python src/diag_global.py grpo_played_b0.1 grpo_played_b0.3_long ...
"""
import sys

import numpy as np
import torch

from common import DEV, dump, to_dev, windows
from policy import load_policy
from rerank import load_cache
from rl import Reward, calibrate


def align(delta: np.ndarray, r: np.ndarray, top: int = 100) -> float:
    """Mean within-context Pearson correlation between the policy change and the reward (top-`top` candidates)."""
    d, rr = delta[:, :top], r[:, :top]
    d = d - d.mean(1, keepdims=True)
    rr = rr - rr.mean(1, keepdims=True)
    return float(np.nanmean((d * rr).sum(1) / np.sqrt((d ** 2).sum(1) * (rr ** 2).sum(1))))


def r2_item(delta: np.ndarray, cand: np.ndarray) -> dict:
    d = delta - delta.mean(1, keepdims=True)  # remove per-context shift (normalisation constant)
    V = cand.max() + 1
    s = np.bincount(cand.ravel(), weights=d.ravel(), minlength=V)
    c = np.bincount(cand.ravel(), minlength=V)
    item = np.where(c > 0, s / np.maximum(c, 1), 0)
    pred = item[cand]
    resid = d - pred
    return {"r2_item_constant": float(1 - resid.var() / d.var()), "delta_std": float(d.std())}


@torch.no_grad()
def train_cache(n=2000):
    """Candidates + RM scores for n RL training contexts (train split), same format as the val cache."""
    from common import RES
    from rm import load_rm
    from rl import is_new, rl_contexts
    p = RES.parent / "checkpoints" / "cand_train.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in z.files}
    idx = np.sort(np.random.default_rng(7).choice(rl_contexts(), n, replace=False))
    ref, rm = load_policy("base").eval(), load_rm("rm0")
    C, P, NEW = [], [], []
    for i in range(0, n, 64):
        W = to_dev(windows(idx[i:i + 64]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c = ref.last_logits(W["tok"].long()).topk(1000).indices
            P.append(torch.sigmoid(rm.score(W, c).float()).cpu().numpy()[None].astype(np.float16))
        C.append(c.cpu().numpy())
        NEW.append(is_new(c, W["tok"].long(), W["mask"]).cpu().numpy())
    z = {"idx": idx, "cand": np.concatenate(C), "P": np.concatenate(P, 1), "new": np.concatenate(NEW)}
    np.savez(p, **z)
    return z


@torch.no_grad()
def main(tags, n=2000, split="val"):
    z = load_cache(1) if split == "val" else train_cache(n)
    idx, cand = z["idx"][:n], z["cand"][:n]
    ref = load_policy("base").eval()
    out = {}
    R = Reward("played", calibrate("rm0"))
    r = R(torch.as_tensor(z["P"][0, :n]).float().to(DEV), torch.as_tensor(cand).long().to(DEV),
          torch.as_tensor(z["new"][:n]).to(DEV)).cpu().numpy()
    out["exact_played (r/beta)"] = r2_item(r, cand)
    for tag in tags:
        pol = load_policy(tag).eval()
        D = []
        for i in range(0, n, 256):
            W = to_dev(windows(idx[i:i + 256]))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lw = pol.last_logits_w(W) if hasattr(pol, "last_logits_w") else pol.last_logits(W["tok"].long())
                lp = torch.log_softmax(lw.float(), -1)
                lr = torch.log_softmax(ref.last_logits(W["tok"].long()).float(), -1)
            c = torch.as_tensor(cand[i:i + 256]).long().to(DEV)
            D.append((lp.gather(1, c) - lr.gather(1, c)).cpu().numpy())
        D = np.concatenate(D)
        out[tag] = r2_item(D, cand)
        out[tag]["corr_with_reward_top100"] = align(D, r)
        out[tag]["corr_with_reward_top1000"] = align(D, r, 1000)
        print(tag, out[tag], flush=True)
    from common import RES
    import json
    prev = RES / f"diag_global_{split}.json"
    if prev.exists():  # merge with earlier runs of other tags
        out = {**json.loads(prev.read_text()), **out}
    dump(f"diag_global_{split}", out)
    print(out)


if __name__ == "__main__":
    split = "train" if sys.argv[1] == "--train" else "val"
    main([t for t in sys.argv[1:] if t != "--train"], split=split)
