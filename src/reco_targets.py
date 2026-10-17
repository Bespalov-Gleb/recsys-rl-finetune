"""Hits on recommendation-driven listens only.

An organic listen (the user picked the track) is something the user would have done anyway, so a "hit" on it says
little about the value of a recommendation. Here relevance is counted only on tracks that in the next 24 hours came
to the user through the service (is_organic = 0) and were listened to >= 50%, all of them and the new ones. This is
still correlational (we do not know what the user would have done with a different list), but it removes the
"the user would have put it on anyway" part that a repeat heuristic exploits most.

python src/reco_targets.py base heur_a2.0_g1.0 exact_played_b0.3 ...
"""
import sys

import numpy as np

from common import RES, dump, events, user_of_event
from evaluate import eval_set

D = 86_400
B = 1000


def target_sets(idx, hs):
    ev = events()
    T, tok, org, pl, sp = (np.asarray(ev[k]) for k in ("t", "tok", "organic", "played", "split"))
    off = np.asarray(ev["offsets"])
    uoe = np.asarray(user_of_event())
    reco, reco_new = [], []
    for i, e in enumerate(idx):
        b = off[uoe[e] + 1]
        j = e + np.searchsorted(T[e:b], T[e] + D, side="right")
        sl = slice(e, j)
        ok = (sp[sl] == sp[e]) & (tok[sl] > 1) & (org[sl] == 0) & (pl[sl] >= 0.5)
        S = np.unique(tok[sl][ok])
        reco.append(S)
        reco_new.append(S[~np.isin(S, hs[i])])
    return reco, reco_new


def per_context(r, sets):
    v = np.full(len(r), np.nan)
    for i, S in enumerate(sets):
        if len(S):
            v[i] = np.isin(r[i], S).sum() / min(10, len(S))
    return v


if __name__ == "__main__":
    names = sys.argv[1:]
    idx, fs, hs = eval_set(2, 3)
    reco, reco_new = target_sets(idx, hs)
    u = np.asarray(user_of_event())[idx]
    users, inv = np.unique(u, return_inverse=True)
    rng = np.random.default_rng(0)
    draws = [np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)) for _ in range(B)]
    vals = {}
    for n in names:
        r = np.load(RES / f"eval2_{n}.npz")["recs"][:, :10]
        vals[n] = {"reco_full": per_context(r, reco), "reco_full_new": per_context(r, reco_new)}
    res = {"n_contexts_with_targets": {k: int((~np.isnan(vals[names[0]][k])).sum()) for k in vals[names[0]]}}
    base = vals[names[0]]
    for n in names:
        res[n] = {}
        for k, v in vals[n].items():
            d = v - base[k]
            ok = ~np.isnan(d)
            s = np.bincount(inv[ok], weights=d[ok], minlength=len(users))
            c = np.bincount(inv[ok], minlength=len(users)).astype(float)
            boots = [(w * s).sum() / max((w * c).sum(), 1) for w in draws]
            res[n][k] = {"value": float(np.nanmean(v)), "diff": float(d[ok].mean()),
                         "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}
        print(n, {k: f"{x['value']:.4f} ({x['diff']:+.4f} [{x['lo']:+.4f}; {x['hi']:+.4f}])" for k, x in res[n].items()}, flush=True)
    print(res["n_contexts_with_targets"])
    dump("reco_targets_2", res)
