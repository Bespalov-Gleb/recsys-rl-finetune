"""Analytic optimum of the KL-regularised objective (no training): pi*(a|s) ~ pi_ref(a|s) * exp(r(s,a) / beta).

For a single recommendation step the action space is enumerable, so the policy that RL converges to can be
written down and evaluated directly: rank candidates by log pi_ref + r / beta. Candidates are the top-N items
of the reference policy (two-stage retrieval -> ranking, N = 1000); items outside keep their pi_ref mass with a
neutral (normalised 0) reward when the KL is computed. beta = inf is the base model, beta = 0 is pure greedy
reward maximisation.

python src/rerank.py cache 1        # score candidates of the val contexts with all reward models
python src/rerank.py sweep 1        # sweep rewards x beta on val
python src/rerank.py full 1         # unrestricted (all 50k items) candidates on a subset of contexts
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from common import DEV, RES, to_dev, windows
from evaluate import eval_set, full_eval
from policy import CKPT, load_policy
from rl import Reward, calibrate, is_new
from rm import load_rm

N = 1000
BETAS = [float("inf"), 3.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0]
PER_USER = {1: 1, 2: 3}


@torch.no_grad()
def cache(split: int, rm_tags=("rm0", "rm1", "rm2"), n_cand: int = N, limit: int | None = None, suffix: str = ""):
    idx = eval_set(split, PER_USER[split])[0][:limit]
    ref = load_policy("base").eval()
    rms = [load_rm(t) for t in rm_tags if (CKPT / f"rm_{t}.pt").exists()]
    C, LP, OUT, P, NEW = [], [], [], [], []
    bs = 64 if n_cand <= 1000 else 4
    for i in range(0, len(idx), bs):
        W = to_dev(windows(idx[i:i + bs]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = ref.last_logits(W["tok"].long())
        lp = torch.log_softmax(lg.float(), -1)
        top = lp.topk(n_cand)
        c = top.indices
        C.append(c.cpu().numpy().astype(np.int32))
        LP.append(top.values.cpu().numpy())
        OUT.append(torch.log1p(-top.values.exp().sum(-1).clamp(max=1 - 1e-6)).cpu().numpy())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(np.stack([torch.sigmoid(m.score(W, c).float()).cpu().numpy().astype(np.float16) for m in rms]))
        NEW.append(is_new(c, W["tok"].long(), W["mask"]).cpu().numpy())
    np.savez(CKPT / f"cand_{split}{suffix}.npz", idx=idx, cand=np.concatenate(C), lp=np.concatenate(LP),
             lp_out=np.concatenate(OUT), P=np.concatenate(P, 1), new=np.concatenate(NEW))


def load_cache(split: int, suffix: str = ""):
    z = np.load(CKPT / f"cand_{split}{suffix}.npz")
    return {k: z[k] for k in z.files}


def rerank(z: dict, reward: str, beta: float, pess: float = 0.0, rm_i: int = 0, norm=None):
    """Top-10 of pi* and its KL to pi_ref."""
    norm = norm or calibrate("rm0")
    R = Reward(reward, norm)
    cand = torch.as_tensor(z["cand"]).long().to(DEV)
    new = torch.as_tensor(z["new"]).to(DEV)
    lp = torch.as_tensor(z["lp"]).float().to(DEV)
    if pess > 0:
        rr = torch.stack([R(torch.as_tensor(z["P"][j]).float().to(DEV), cand, new) for j in range(1, len(z["P"]))])
        r = rr.mean(0) - pess * rr.std(0)
    else:
        r = R(torch.as_tensor(z["P"][rm_i]).float().to(DEV), cand, new)
    if beta == float("inf"):
        score, kl = lp, torch.zeros(len(lp), device=DEV)
    else:
        b = max(beta, 1e-9)
        score = lp + r / b if beta > 0 else r + 1e-6 * lp
        if beta > 0:
            lo = torch.as_tensor(z["lp_out"]).float().to(DEV)
            logits = torch.cat([lp + r / b, lo[:, None]], 1)
            logZ = torch.logsumexp(logits, 1)
            pstar = torch.exp(lp + r / b - logZ[:, None])
            kl = (pstar * (r / b)).sum(1) - logZ  # E_pi*[r/b] - log Z (outside mass has r = 0)
        else:
            kl = torch.full((len(lp),), float("nan"), device=DEV)
    top = score.topk(10).indices
    recs = torch.gather(cand, 1, top).cpu().numpy()
    exp_r = (torch.gather(r, 1, top)).mean().item()
    return recs, {"kl_ref": float(kl.mean().item()), "proxy_r_top10": exp_r}


def sweep(split: int, rewards=("played", "seconds", "full", "like", "played_db", "lt", "paper2"), betas=BETAS,
          pess_list=(0.0,), suffix=""):
    z = load_cache(split, suffix)
    idx = z["idx"]
    from rm import load_rm as _l
    rms = [_l("rm0")]
    for rw in rewards:
        for b in betas:
            for ps in pess_list:
                if b == float("inf") and (rw != rewards[0] or ps > 0):
                    continue
                name = ("base" + suffix) if b == float("inf") else f"exact_{rw}_b{b}" + (f"_p{ps}" if ps else "") + suffix
                recs, d = rerank(z, rw, b, pess=ps)
                full_recs = np.concatenate([recs, z["cand"][:, :100]], 1)  # HR@100 slot: keep base order after top-10
                res = full_eval(name, z["cand"][:, :100] if b == float("inf") else _dedup(full_recs),
                                split=split, per_user=PER_USER[split], extra=d, rms=rms, save_arrays=True,
                                limit=len(idx))
                print(name, {k: round(res[k], 4) for k in ("HR@10", "Recall@10_full", "proxy_played", "gold_played",
                                                          "gold_full", "mean_len_s", "repeat_share", "coverage", "kl_ref")},
                      flush=True)


def _dedup(r: np.ndarray, K: int = 100) -> np.ndarray:
    out = np.zeros((len(r), K), np.int64)
    for i, row in enumerate(r):
        _, first = np.unique(row, return_index=True)
        out[i] = row[np.sort(first)][:K]
    return out


if __name__ == "__main__":
    cmd, split = sys.argv[1], int(sys.argv[2])
    if cmd == "cache":
        cache(split)
    elif cmd == "sweep":
        sweep(split)
    elif cmd == "strict":  # robustness: same test protocol, histories strictly earlier in time (RS_STRICT=1)
        cache(split, suffix="_strict")
        sweep(split, rewards=("played", "full", "like"), betas=[float("inf"), 1.0, 0.3, 0.1], suffix="_strict")
    elif cmd == "testsweep":
        sweep(split, betas=[float("inf"), 1.0, 0.3, 0.1, 0.0])
    elif cmd == "pess":
        sweep(split, rewards=("played", "full"), betas=[0.3, 0.1, 0.05, 0.0], pess_list=(0.0, 1.0, 3.0))
    elif cmd == "full":
        cache(split, rm_tags=("rm0",), n_cand=50_000 - 1, limit=1000, suffix="_fullvocab")
        sweep(split, rewards=("played", "full", "seconds"), betas=[float("inf"), 0.3, 0.1, 0.05, 0.0], suffix="_fullvocab")
