"""Paired bootstrap over users: difference of a model vs a reference on the test contexts, 95% CI.

Per-context metrics are recomputed from the saved top-10 lists (results/eval2_<name>.npz).
python src/bootstrap.py base exact_played_b0.3 grpo_played_b0.3 ...
"""
import sys

import numpy as np

from common import dump, events, user_of_event
from evaluate import eval_set

B = 1000


def per_context(name, split=2, per_user=3):
    z = np.load(f"{__import__('common').RES}/eval{split}_{name}.npz")
    idx, fs, hs = eval_set(split, per_user)
    r = z["recs"]
    nxt = events()["tok"][idx]
    out = {"HR@10": (r == nxt[:, None]).any(1).astype(float)}
    for key in ("all", "full", "skip", "like"):
        v = np.full(len(idx), np.nan)
        for i, S in enumerate(fs[key]):
            if len(S):
                v[i] = np.isin(r[i], S).sum() / min(10, len(S))
        out[f"Recall@10_{key}"] = v
    # discoveries: listen+ targets the user had never listened to before
    v = np.full(len(idx), np.nan)
    for i, S in enumerate(fs["full"]):
        new = S[~np.isin(S, hs[i])]
        if len(new):
            v[i] = np.isin(r[i], new).sum() / min(10, len(new))
    out["Recall@10_full_new"] = v
    if "gold" in z.files:
        out["gold_played"] = z["gold"][:, 0]
        out["gold_full"] = z["gold"][:, 1]
        out["gold_like"] = z["gold"][:, 2]
    if "proxy" in z.files:
        out["proxy_played"] = z["proxy"][:, 0]
    return idx, out


def compare(ref, names, split=2):
    idx, base = per_context(ref, split)
    u = user_of_event()[idx]
    users, inv = np.unique(u, return_inverse=True)
    rng = np.random.default_rng(0)
    draws = [np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)) for _ in range(B)]
    res = {}
    for n in names:
        _, m = per_context(n, split)
        res[n] = {}
        for k, v in m.items():
            d = v - base[k]
            ok = ~np.isnan(d)
            # user-level sums so that a bootstrap draw re-weights whole users
            s = np.bincount(inv[ok], weights=d[ok], minlength=len(users))
            c = np.bincount(inv[ok], minlength=len(users)).astype(float)
            boots = [(w * s).sum() / max((w * c).sum(), 1) for w in draws]
            res[n][k] = {"diff": float(d[ok].mean()), "lo": float(np.percentile(boots, 2.5)),
                         "hi": float(np.percentile(boots, 97.5)), "value": float(v[~np.isnan(v)].mean())}
    return res


if __name__ == "__main__":
    out = None
    if sys.argv[1].startswith("--out="):
        out = sys.argv.pop(1)[len("--out="):]
    ref, names = sys.argv[1], sys.argv[2:]
    res = compare(ref, names)
    dump(out or f"bootstrap_vs_{ref}", res)
    for n, d in res.items():
        print(n, {k: f"{v['diff']:+.4f} [{v['lo']:+.4f}; {v['hi']:+.4f}]" for k, v in d.items()})
