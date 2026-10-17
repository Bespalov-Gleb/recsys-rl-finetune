"""Repeats vs discoveries: split the next-24-hour relevance into tracks the user has already listened to and new ones.

For every saved top-10 (results/eval<split>_<name>.npz):
  Recall@10_full_known / Recall@10_full_new  -- listen+ targets inside / outside the user's history
  Recall@10_skip_known                         -- skipped targets the user already knew
  selectivity = Recall listen+ / Recall skips  (does the list prefer what will be finished over what will be skipped)
python src/novelty.py 2 base heur_a2.0_g1.0 cross_played_b0.1 ...
"""
import sys

import numpy as np

from common import RES, dump
from evaluate import eval_set


def per_context(name, split):
    z = np.load(RES / f"eval{split}_{name}.npz")
    idx, fs, hs = eval_set(split, {1: 1, 2: 3}[split])
    r = z["recs"][:, :10]
    out = {k: [] for k in ("full_known", "full_new", "skip_known", "full", "skip")}
    for i in range(len(r)):
        for key in ("full", "skip"):
            S = fs[key][i]
            if len(S):
                out[key].append(np.isin(r[i], S).sum() / min(10, len(S)))
            known = S[np.isin(S, hs[i])]
            new = S[~np.isin(S, hs[i])]
            if key == "full" and len(new):
                out["full_new"].append(np.isin(r[i], new).sum() / min(10, len(new)))
            if len(known) and key + "_known" in out:
                out[key + "_known"].append(np.isin(r[i], known).sum() / min(10, len(known)))
    res = {f"Recall@10_{k}": float(np.mean(v)) for k, v in out.items()}
    res["selectivity_full_over_skip"] = res["Recall@10_full"] / res["Recall@10_skip"]
    res["new_in_list"] = float(np.mean([1 - np.isin(r[i], hs[i]).mean() for i in range(len(r))]))
    return res


if __name__ == "__main__":
    split, names = int(sys.argv[1]), sys.argv[2:]
    res = {n: per_context(n, split) for n in names}
    dump(f"novelty_{split}", res)
    for n, d in res.items():
        print(n, {k: round(v, 4) for k, v in d.items()})
