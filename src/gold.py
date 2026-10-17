""""Gold" reward model (Gao et al. 2022 methodology): an independent judge used only for evaluation.

Deliberately different from the proxy RM that RL optimizes: gradient boosting (LightGBM) instead of a neural net,
a 5x longer history (1000 events), target-encoded item / artist statistics and recency in seconds. Item
statistics are computed on train events before T_STAT; gold training examples are sampled after T_STAT
(still inside train), so the statistics never contain the example's own label.

python src/gold.py train
python src/gold.py eval
"""
from __future__ import annotations

import sys
import time

import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import DEV, RES, UNK, dump, events, meta, to_dev, user_of_event, vocab, windows
from rm import HEADS, cross_feats, heldout_idx, label_tensors, metrics_on

LW = 1000
CKPT = RES.parent / "checkpoints"
D = 86_400


def item_stats() -> np.ndarray:
    """(V, 7): n_nonorg, mean played, full rate, skip rate, like rate, artist mean played, artist n (log)."""
    p = CKPT / "gold_item_stats.npy"
    if p.exists():
        return np.load(p)
    ev = events()
    voc = vocab()
    V = len(voc["tok_len"])
    t_stat = meta()["t_val"] - 30 * D
    s = (ev["t"] < t_stat) & (ev["organic"] == 0) & (ev["tok"] > UNK)
    tok, pl_ = ev["tok"][s], ev["played"][s]
    lk = np.maximum(ev["like"][s], 0)
    n = np.bincount(tok, minlength=V).astype(float)
    prior = 20.0
    gm, gf, gs, gl = pl_.mean(), (pl_ >= .5).mean(), (pl_ < .1).mean(), lk.mean()
    sm = lambda x, g: (np.bincount(tok, weights=x, minlength=V) + prior * g) / (n + prior)
    art = voc["tok_artist"]
    na = np.bincount(art[tok], minlength=art.max() + 1).astype(float)
    am = (np.bincount(art[tok], weights=pl_, minlength=art.max() + 1) + prior * gm) / (na + prior)
    out = np.stack([np.log1p(n), sm(pl_, gm), sm((pl_ >= .5).astype(float), gf), sm((pl_ < .1).astype(float), gs),
                    sm(lk.astype(float), gl), am[art], np.log1p(na[art])], 1).astype(np.float32)
    np.save(p, out)
    return out


FEATS = ["n_i", "m_i", "has_i", "n_a", "m_a", "rec_i", "same_last", "art_share",
         "it_n", "it_played", "it_full", "it_skip", "it_like", "art_played", "art_n",
         "since_last_s", "u_played", "u_skip", "u_org", "u_n", "u_played_rec", "u_full_rec10", "len", "art_last10"]


@torch.no_grad()
def gold_feats(e: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """e (B,) decision events (history = events before e), cand (B,C) tokens -> (B*C, F)."""
    st = torch.as_tensor(item_stats()).to(DEV)
    voc = vocab()
    art_of = torch.as_tensor(voc["tok_artist"]).to(DEV).long()
    tl = torch.as_tensor(voc["tok_len"]).to(DEV)
    out = []
    for i in range(0, len(e), 64):
        W = to_dev(windows(e[i:i + 64], length=LW, keys=("tok", "played", "organic", "t")))
        tok, pl_, org, mask, t = W["tok"].long(), W["played"].float(), W["organic"].float(), W["mask"], W["t"].double()
        c = torch.as_tensor(cand[i:i + 64]).to(DEV).long()
        B, C = c.shape
        allowed = mask[:, None, :].expand(-1, C, -1)
        q = torch.full(c.shape, float(LW), device=DEV)
        cr = cross_feats(c, q, tok, pl_, art_of, allowed)
        eq = (c[:, :, None] == tok[:, None, :]) & allowed
        t_ref = t[:, -1:]
        last_t = torch.where(eq, t[:, None, :], torch.full_like(t[:, None, :], -1)).amax(-1)
        since = torch.where(last_t >= 0, torch.log1p((t_ref - last_t).clamp(min=0)), torch.full_like(last_t, 20.0)).float()
        mf = mask.float()
        n = mf.sum(1, keepdim=True).clamp(min=1)
        rec = mf * (1 - org)
        u = torch.stack([(pl_ * mf).sum(1) / n[:, 0], ((pl_ < .1).float() * mf).sum(1) / n[:, 0],
                         (org * mf).sum(1) / n[:, 0], torch.log1p(n[:, 0]),
                         (pl_ * rec).sum(1) / rec.sum(1).clamp(min=1),
                         ((pl_[:, -10:] >= .5).float() * rec[:, -10:]).sum(1) / rec[:, -10:].sum(1).clamp(min=1)], 1)
        a10 = ((art_of[c][:, :, None] == art_of[tok[:, -10:]][:, None, :]) & mask[:, None, -10:]).float().sum(-1)
        f = torch.cat([cr, st[c], since[..., None], u[:, None, :].expand(-1, C, -1),
                       torch.log(tl[c].clamp(min=10))[..., None], a10[..., None]], -1)
        out.append(f.reshape(B * C, -1).cpu().numpy())
    return np.concatenate(out)


def train():
    ev = events()
    t_stat = meta()["t_val"] - 30 * D
    ok = np.flatnonzero((ev["split"] == 0) & (ev["t"] >= t_stat) & (ev["organic"] == 0) & (ev["tok"] > UNK))
    off = ev["offsets"]
    ok = ok[ok - off[user_of_event()[ok]] >= 20]
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(ok, min(1_500_000, len(ok)), replace=False))
    va = heldout_idx(1, 100_000)
    t0 = time.time()
    Xtr = gold_feats(tr, ev["tok"][tr][:, None])
    Xva = gold_feats(va, ev["tok"][va][:, None])
    print("features", Xtr.shape, time.time() - t0, flush=True)
    ltr, lva = label_tensors(tr), label_tensors(va)
    log = {}
    for k in HEADS:
        ytr, yva = ltr[k].astype(float), lva[k].astype(float)
        mtr, mva = ytr >= 0, yva >= 0
        obj = "cross_entropy"
        params = dict(objective=obj, learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
        dtr = lgb.Dataset(Xtr[mtr], ytr[mtr], feature_name=FEATS)
        dva = lgb.Dataset(Xva[mva], yva[mva], reference=dtr)
        b = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        b.save_model(str(CKPT / f"gold_{k}.txt"))
        log[k] = {"iters": b.best_iteration, "val": b.best_score["valid_0"]["cross_entropy"]}
        print(k, log[k], flush=True)
    dump("train_gold", log)


_boosters = None


def boosters():
    global _boosters
    if _boosters is None:
        _boosters = [lgb.Booster(model_file=str(CKPT / f"gold_{k}.txt")) for k in HEADS]
    return _boosters


def gold_score(e: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """(B,C,5) probabilities from the gold judge."""
    X = gold_feats(e, cand)
    P = np.stack([b.predict(X, num_threads=8) for b in boosters()], -1)
    return P.reshape(cand.shape[0], cand.shape[1], len(HEADS))


def evaluate():
    idx = np.load(RES / "rm_test_idx.npy")
    P = gold_score(idx, events()["tok"][idx][:, None])[:, 0]
    res = metrics_on(P, idx)
    print(res)
    np.save(RES / "rm_pred_test_gold.npy", P)
    dump("rm_eval_gold", res)
    imp = {k: dict(zip(FEATS, b.feature_importance("gain").round(0).tolist())) for k, b in zip(HEADS, boosters())}
    dump("gold_importance", imp)


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1]]()
