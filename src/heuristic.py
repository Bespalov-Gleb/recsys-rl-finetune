"""Control for the cross-feature policy: a hand-made re-ranker over the same reference top-1000 without RM or RL.

score = log pi_ref(a|s) + alpha * log1p(plays of a in the history) + gamma * (mean played ratio of a in the history)
Grid over (alpha, gamma) on validation; the best few configurations are then evaluated on test. If this simple
"repeat what the user finished listening to" rule matches the RL-trained cross head, the head has learned a repeat
heuristic rather than something only the reward model knows.

python src/heuristic.py val       # grid on validation
python src/heuristic.py test A G  # one configuration on test
"""
import sys

import numpy as np
import torch

from common import DEV, to_dev, vocab, windows
from evaluate import full_eval
from rerank import PER_USER, load_cache
from rm import cross_feats, load_rm


@torch.no_grad()
def features(split):
    z = load_cache(split)
    idx, cand = z["idx"], z["cand"]
    art_of = torch.as_tensor(vocab()["tok_artist"]).long().to(DEV)
    n_i, m_i = [], []
    for i in range(0, len(idx), 64):
        W = to_dev(windows(idx[i:i + 64]))
        c = torch.as_tensor(cand[i:i + 64]).long().to(DEV)
        allowed = W["mask"][:, None, :].expand(-1, c.shape[1], -1)
        q = torch.full(c.shape, float(W["tok"].shape[1]), device=DEV)
        cr = cross_feats(c, q, W["tok"].long(), W["played"].float(), art_of, allowed)
        n_i.append(cr[..., 0].cpu().numpy())  # log1p(count)
        m_i.append(cr[..., 1].cpu().numpy())  # mean played ratio (0 if never)
    return z, np.concatenate(n_i), np.concatenate(m_i)


def run(split, configs):
    z, n_i, m_i = features(split)
    rm = load_rm("rm0")
    out = {}
    for a, g in configs:
        score = z["lp"] + a * n_i + g * m_i
        top = np.argsort(-score, 1)[:, :100]
        recs = np.take_along_axis(z["cand"], top, 1)
        name = f"heur_a{a}_g{g}"
        r = full_eval(name, recs, split=split, per_user=PER_USER[split], rms=[rm], limit=len(z["idx"]))
        out[name] = r
        print(name, {k: round(r[k], 4) for k in ("HR@10", "NDCG@10_all", "Recall@10_full", "Recall@10_skip",
                                                  "gold_played", "repeat_share", "coverage")}, flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "val":
        grid = [(a, g) for a in (0.0, 0.5, 1.0, 2.0) for g in (0.0, 1.0, 2.0, 4.0) if (a, g) != (0.0, 0.0)]
        run(1, grid)
    else:
        run(2, [(float(sys.argv[2]), float(sys.argv[3]))])
