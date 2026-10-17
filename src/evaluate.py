"""One evaluation protocol for every recommender (base policy, RL policies, analytic rerankers).

For each evaluation context (a listen event in val/test; history = everything before it):
  * relevance: next-item HR/NDCG@10, HR@100; Recall/NDCG/MAP@10 against the items listened in the next 24 h,
    separately for all / listen+ (>=50%) / skipped (<10%) / liked / adopted tracks;
  * beyond accuracy: coverage, Gini, top-1 share, pairwise Jaccard between users, repeat share, popularity,
    track length, artists per list;
  * rewards of the top-10 list: proxy RM (the one RL optimizes) and the independent gold judge (LightGBM);
  * distribution: entropy and KL(pi || pi_ref) when a distribution is available.
"""
from __future__ import annotations

import pickle

import numpy as np
import torch

from common import DEV, RES, dump, eval_points, events, future_sets, history_sets, ranking_metrics, to_dev, vocab, windows
from rm import HEADS

CACHE = RES.parent / "checkpoints"


def eval_set(split: int, per_user: int):
    p = CACHE / f"evalset_{split}_{per_user}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    idx = eval_points(split, per_user=per_user)
    obj = (idx, future_sets(idx), history_sets(idx))
    p.write_bytes(pickle.dumps(obj))
    return obj


@torch.no_grad()
def policy_logits(policy, idx: np.ndarray, bs: int = 256):
    """Yields (slice, logits (b,V) float32) over the evaluation contexts."""
    policy.eval()
    for i in range(0, len(idx), bs):
        W = to_dev(windows(idx[i:i + bs]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = policy.last_logits_w(W) if hasattr(policy, "last_logits_w") else policy.last_logits(W["tok"].long())
        yield slice(i, i + bs), lg.float()


@torch.no_grad()
def policy_topk(policy, idx: np.ndarray, K: int = 100, ref=None):
    recs, ent, kl = [], [], []
    ref_it = policy_logits(ref, idx) if ref is not None else None
    for sl, lg in policy_logits(policy, idx):
        lp = torch.log_softmax(lg, -1)
        recs.append(lg.topk(K).indices.cpu().numpy())
        ent.append(-(lp.exp() * lp).sum(-1).cpu().numpy())
        if ref_it is not None:
            _, lr = next(ref_it)
            lr = torch.log_softmax(lr, -1)
            kl.append((lp.exp() * (lp - lr)).sum(-1).cpu().numpy())
    out = {"entropy": float(np.concatenate(ent).mean())}
    if kl:
        out["kl_ref"] = float(np.concatenate(kl).mean())
    return np.concatenate(recs), out


@torch.no_grad()
def proxy_list_scores(rms, idx: np.ndarray, recs: np.ndarray, bs: int = 256) -> np.ndarray:
    """(n_rm, n, K, 5) probabilities of every RM in `rms` for every recommended item."""
    out = []
    for m in rms:
        m.eval()
        res = []
        for i in range(0, len(idx), bs):
            W = to_dev(windows(idx[i:i + bs]))
            c = torch.as_tensor(recs[i:i + bs]).to(DEV).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                res.append(torch.sigmoid(m.score(W, c).float()).cpu().numpy())
        out.append(np.concatenate(res))
    return np.stack(out)


def reward_summary(P: np.ndarray, recs: np.ndarray, prefix: str) -> dict:
    """P (n,K,5) probabilities -> mean per head over the list, expected listened seconds, adopt for new items."""
    voc = vocab()
    res = {f"{prefix}_{h}": float(P[..., i].mean()) for i, h in enumerate(HEADS)}
    res[f"{prefix}_seconds"] = float((P[..., 0] * voc["tok_len"][recs]).mean())
    return res


def full_eval(name: str, recs: np.ndarray, split: int = 2, per_user: int = 3, extra: dict | None = None,
              rms=None, gold: bool = True, save_arrays: bool = True, limit: int | None = None) -> dict:
    from gold import gold_score
    idx, fs, hs = eval_set(split, per_user)
    if limit is not None:  # first `limit` contexts only (expensive variants)
        idx, hs = idx[:limit], hs[:limit]
        fs = {k: v[:limit] for k, v in fs.items()}
    r10 = recs[:, :10]
    res = {"name": name, "split": split, "n": len(idx)}
    res.update(ranking_metrics(recs, idx, fs, hs))
    arrays = {"hit10": (r10 == events()["tok"][idx][:, None]).any(1)}
    if rms:
        P = proxy_list_scores(rms, idx, r10)
        res.update(reward_summary(P[0], r10, "proxy"))
        if len(P) > 1:
            res.update(reward_summary(P.mean(0), r10, "ens"))
            res["ens_std_played"] = float(P[..., 0].std(0).mean())
        arrays["proxy"] = P[0].mean(1)
    if gold:
        G = gold_score(idx, r10)
        res.update(reward_summary(G, r10, "gold"))
        arrays["gold"] = G.mean(1)
    if extra:
        res.update(extra)
    tag = f"eval{split}_{name}"
    dump(tag, res)
    if save_arrays:
        np.savez_compressed(RES / f"{tag}.npz", recs=r10.astype(np.int32), **arrays)
    return res
