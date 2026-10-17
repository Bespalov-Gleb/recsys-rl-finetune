"""Reward model diagnostics.

1. Held-out quality by slice (test split, logged recommendation listens): track length, first listen vs repeat,
   popularity; proxy RM vs gold judge vs simple baselines.
2. Off-policy agreement: on val contexts, the proxy RMs and the gold judge score the top-1000 candidates of the
   reference policy. Logged data only cover items the service actually showed, so this is where the RM has to
   extrapolate. We measure, per rank bucket of pi_ref, the proxy-gold correlation and the ensemble disagreement.

python src/rm_analysis.py
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import RES, dump, events, vocab
from gold import gold_score
from rerank import load_cache
from rm import HEADS, label_tensors


def sliced():
    idx = np.load(RES / "rm_test_idx.npy")
    ev = events()
    voc = vocab()
    lab = label_tensors(idx)
    preds = {t: np.load(RES / f"rm_pred_test_{t}.npy") for t in ("rm0", "rm1", "rm2", "gold") if (RES / f"rm_pred_test_{t}.npy").exists()}
    tok = ev["tok"][idx]
    ln = voc["tok_len"][tok]
    pop = voc["tok_pop"][tok]
    slices = {
        "len<150s": ln < 150, "len150-240s": (ln >= 150) & (ln < 240), "len>=240s": ln >= 240,
        "first_listen": ev["first"][idx] == 1, "repeat": ev["first"][idx] == 0,
        "pop_top5k": tok < 5002, "pop_5k-20k": (tok >= 5002) & (tok < 20002), "pop_20k-50k": tok >= 20002,
    }
    y_full, y_pl = lab["full"], lab["played"]
    out = {}
    for name, s in slices.items():
        r = {"n": int(s.sum()), "full_rate": float(y_full[s].mean()), "played_mean": float(y_pl[s].mean())}
        for t, P in preds.items():
            r[f"{t}_full_auc"] = float(roc_auc_score(y_full[s], P[s, 1]))
            r[f"{t}_played_bias"] = float(P[s, 0].mean() - y_pl[s].mean())
            r[f"{t}_played_mae"] = float(np.abs(P[s, 0] - y_pl[s]).mean())
        out[name] = r
    # correlation between proxy and gold on logged pairs
    if "gold" in preds:
        out["logged_pairs_proxy_gold_spearman_played"] = float(spearmanr(preds["rm0"][:, 0], preds["gold"][:, 0])[0])
    return out


def off_policy(n_ctx: int = 300):
    z = load_cache(1)
    idx, cand, P = z["idx"][:n_ctx], z["cand"][:n_ctx], z["P"][:, :n_ctx].astype(np.float32)
    G = gold_score(idx, cand)  # (n, 1000, 5)
    buckets = {"rank1-10": slice(0, 10), "rank11-100": slice(10, 100), "rank101-300": slice(100, 300),
               "rank301-1000": slice(300, 1000)}
    out = {}
    for h, hi in (("played", 0), ("full", 1), ("like", 2)):
        for b, sl in buckets.items():
            rho = [spearmanr(P[0, i, sl, hi], G[i, sl, hi])[0] for i in range(len(idx))]
            r = {"proxy_gold_spearman": float(np.nanmean(rho)),
                 "proxy_mean": float(P[0, :, sl, hi].mean()), "gold_mean": float(G[:, sl, hi].mean())}
            if P.shape[0] > 1:
                r["ens_std"] = float(P[:, :, sl, hi].std(0).mean())
                r["rm0_vs_rm1_spearman"] = float(np.nanmean([spearmanr(P[0, i, sl, hi], P[1, i, sl, hi])[0] for i in range(len(idx))]))
            out[f"{h}/{b}"] = r
        # where does the proxy's argmax land, and does gold agree?
        am = P[0, :, :, hi].argmax(1)
        out[f"{h}/proxy_argmax"] = {"mean_rank": float(am.mean()), "median_rank": float(np.median(am)),
                                   "proxy_at_argmax": float(P[0, np.arange(len(am)), am, hi].mean()),
                                   "gold_at_argmax": float(G[np.arange(len(am)), am, hi].mean()),
                                   "gold_at_rank1": float(G[:, 0, hi].mean()),
                                   "gold_max_possible": float(G[:, :, hi].max(1).mean())}
    np.savez_compressed(RES / "offpolicy_scores.npz", P=P[:, :, :, :3].astype(np.float16), G=G[..., :3].astype(np.float16))
    return out


if __name__ == "__main__":
    res = {"sliced": sliced(), "off_policy": off_policy()}
    dump("rm_analysis", res)
    print(res)
