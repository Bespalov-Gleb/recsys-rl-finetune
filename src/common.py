"""Shared data access: event arrays, context windows, evaluation points and ranking / beyond-accuracy metrics."""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L = 200          # context length (events) for both the policy and the reward model
# RS_STRICT=1: evaluation histories contain only events strictly earlier in time (see strict_end)
STRICT = os.environ.get("RS_STRICT", "0") == "1"
PAD, UNK = 0, 1


@lru_cache(None)
def meta() -> dict:
    return json.loads((DATA / "meta.json").read_text())


@lru_cache(None)
def events() -> dict[str, np.ndarray]:
    """Memory-mapped per-column arrays (data/ev/*.npy): several processes share the OS page cache."""
    return {p.stem: np.load(p, mmap_mode="r") for p in sorted((DATA / "ev").glob("*.npy"))}


@lru_cache(None)
def vocab() -> dict[str, np.ndarray]:
    z = np.load(DATA / "vocab.npz")
    return {k: z[k] for k in z.files}


@lru_cache(None)
def user_of_event() -> np.ndarray:
    return events()["uid_idx"]


def strict_end(idx: np.ndarray) -> np.ndarray:
    """First event of the same user with the same timestamp as idx. Yambda time is on a 5-second grid and 19 % of
    consecutive listens share a timestamp (fast skipping), so their true order is unknown; a history that ends here
    contains only events strictly earlier in time."""
    ev = events()
    t = ev["t"]
    u0 = ev["offsets"][user_of_event()[idx]]
    out = np.asarray(idx).copy()
    for i, (e, a) in enumerate(zip(idx, u0)):
        out[i] = a + np.searchsorted(t[a:e + 1], t[e], side="left")
    return out


def windows(idx: np.ndarray, length: int = L, include_self: bool = False,
            keys: tuple = ("tok", "played", "organic"), strict: bool = False) -> dict[str, np.ndarray]:
    """Left-padded windows of the `length` events BEFORE each event index (optionally including it).

    Returns arrays (n, length): tok, played, organic, mask (1 = real event). Likes are NOT part of the history:
    a like of an earlier track can happen after the decision moment, so it would leak the future."""
    ev = events()
    u0 = ev["offsets"][user_of_event()[idx]]
    strict = strict or (STRICT and not include_self)
    end = (strict_end(idx) if strict else idx) + (1 if include_self else 0)
    pos = end[:, None] - length + np.arange(length)[None, :]
    mask = pos >= u0[:, None]
    pos = np.where(mask, pos, 0)
    out = {"mask": mask}
    for k in keys:
        a = ev[k][pos]
        out[k] = np.where(mask, a, 0)
    out["tok"] = np.where(mask, out["tok"], PAD)
    return out


def eval_points(split: int, per_user: int = 4, min_hist: int = 20, seed: int = 0) -> np.ndarray:
    """Random listen events (known-vocabulary target) inside a split, up to `per_user` per user."""
    ev = events()
    rng = np.random.default_rng(seed + split)
    off = ev["offsets"]
    out = []
    for u in range(len(off) - 1):
        a, b = off[u], off[u + 1]
        idx = np.arange(a, b)
        sel = idx[(ev["split"][a:b] == split) & (ev["tok"][a:b] > UNK) & (idx - a >= min_hist)]
        if len(sel):
            out.append(rng.choice(sel, min(per_user, len(sel)), replace=False))
    return np.sort(np.concatenate(out))


def future_sets(idx: np.ndarray, horizon: int = 86_400) -> dict[str, list[np.ndarray]]:
    """Items the user listens to from the evaluation event on, within `horizon` seconds and the same split.

    Keys: all, full (>=50% played), skip (<10% played), like (liked), adopt (a recommended new track that the user
    then re-listened to on their own / liked within 3 days)."""
    ev = events()
    uoe = user_of_event()
    off = ev["offsets"]
    res = {k: [] for k in ("all", "full", "skip", "like", "adopt")}
    for e in idx:
        u = uoe[e]
        b = off[u + 1]
        t0, s = ev["t"][e], ev["split"][e]
        j = e + np.searchsorted(ev["t"][e:b], t0 + horizon, side="right")
        sl = slice(e, j)
        ok = (ev["split"][sl] == s) & (ev["tok"][sl] > UNK)
        tok, pl_ = ev["tok"][sl][ok], ev["played"][sl][ok]
        res["all"].append(np.unique(tok))
        res["full"].append(np.unique(tok[pl_ >= 0.5]))
        res["skip"].append(np.unique(tok[pl_ < 0.1]))
        res["like"].append(np.unique(tok[ev["like"][sl][ok] == 1]))
        res["adopt"].append(np.unique(tok[ev["adopt"][sl][ok] == 1]))
    return res


def history_sets(idx: np.ndarray) -> list[np.ndarray]:
    """All tokens the user has listened to before the evaluation event (whole history, not just the window)."""
    ev = events()
    off = ev["offsets"]
    uoe = user_of_event()
    return [np.unique(ev["tok"][off[uoe[e]]:e]) for e in idx]


def _dcg(rel: np.ndarray) -> float:
    return float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())


def ranking_metrics(recs: np.ndarray, idx: np.ndarray, fsets: dict, hsets: list, K: int = 10) -> dict:
    """recs: (n, >=K) recommended tokens, best first."""
    ev = events()
    voc = vocab()
    nxt = ev["tok"][idx]
    r = recs[:, :K]
    hit_pos = (r == nxt[:, None])
    hr = hit_pos.any(1)
    ndcg_next = (hit_pos / np.log2(np.arange(2, K + 2))[None, :]).sum(1)
    out = {f"HR@{K}": hr.mean(), f"NDCG@{K}": ndcg_next.mean()}
    if recs.shape[1] >= 100:
        out["HR@100"] = (recs[:, :100] == nxt[:, None]).any(1).mean()
    for key, sets in fsets.items():
        rec_, nd_, ap_, n_ = [], [], [], 0
        for i, S in enumerate(sets):
            if len(S) == 0:
                continue
            rel = np.isin(r[i], S).astype(float)
            m = min(K, len(S))
            rec_.append(rel.sum() / m)
            nd_.append(_dcg(rel) / _dcg(np.ones(m)))
            prec = np.cumsum(rel) / np.arange(1, K + 1)
            ap_.append((prec * rel).sum() / m)
        out[f"Recall@{K}_{key}"] = float(np.mean(rec_))
        out[f"NDCG@{K}_{key}"] = float(np.mean(nd_))
        out[f"MAP@{K}_{key}"] = float(np.mean(ap_))
    # beyond accuracy
    flat = r.ravel()
    cnt = np.bincount(flat, minlength=len(voc["tok_len"]))[2:]
    out["coverage"] = float((cnt > 0).mean())
    sc = np.sort(cnt)
    n = len(sc)
    out["gini"] = float(1 - 2 * ((n - np.arange(1, n + 1) + 0.5) * sc).sum() / (n * sc.sum()))
    out["top1_share"] = float(sc[-1] / len(r))  # share of lists that contain the most recommended item
    out["repeat_share"] = float(np.mean([np.isin(r[i], hsets[i]).mean() for i in range(len(r))]))
    out["mean_len_s"] = float(voc["tok_len"][r].mean())
    out["mean_logpop"] = float(np.log1p(voc["tok_pop"][r]).mean())
    out["artists_per_list"] = float(np.mean([len(np.unique(voc["tok_artist"][x])) for x in r]))
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(r), 3000), rng.integers(0, len(r), 3000)
    jac = [len(np.intersect1d(r[i], r[j])) / len(np.union1d(r[i], r[j])) for i, j in zip(a, b) if i != j]
    out["pair_jaccard"] = float(np.mean(jac))
    return {k: float(v) for k, v in out.items()}


def to_dev(w: dict) -> dict:
    return {k: torch.as_tensor(v).to(DEV) for k, v in w.items()}


def dump(name: str, obj) -> None:
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def cosine_lr(step: int, total: int, base: float, warm: int = 200) -> float:
    if step < warm:
        return base * step / warm
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
