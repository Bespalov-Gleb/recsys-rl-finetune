"""Softmax-bottleneck check: can the per-context reward shift r(s, .) be written as <w_s, e_a> + b_s with the
policy's own item embeddings e_a (what a GPT-2 head with tied embeddings can express by changing only the
context vector)? Per context: fit w_s on a random half of the reference top-1000 candidates (ridge), measure R^2 on
the other half. Compared with the same fit for the reference log-probabilities themselves (which the head does
express) and for a shuffled reward (chance level).

python src/bottleneck.py
"""
import numpy as np
import torch

from common import DEV, dump
from policy import CKPT
from rerank import load_cache
from rl import Reward, calibrate


def heldout_r2(E, y, rng, lam=1.0):
    n = len(y)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]
    mu_x, mu_y = E[tr].mean(0), y[tr].mean()  # centring = unpenalised intercept
    X = E - mu_x
    A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, X[tr].T @ (y[tr] - mu_y))
    p = X[te] @ w + mu_y
    return 1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()


def main(n_ctx=300):
    z = load_cache(1)
    E = torch.load(CKPT / "policy_base.pt", map_location="cpu")["sd"]["gpt.wte.weight"].float().numpy()
    R = Reward("played", calibrate("rm0"))
    rng = np.random.default_rng(0)
    lams = (0.1, 1.0, 10.0, 100.0)
    out = {f"{k}_lam{l:g}": [] for k in ("reward", "reward_shuffled", "ref_logprob") for l in lams}
    for i in range(n_ctx):
        c = z["cand"][i]
        r = R(torch.as_tensor(z["P"][0, i:i + 1]).float().to(DEV), torch.as_tensor(c[None]).long().to(DEV),
              torch.as_tensor(z["new"][i:i + 1]).to(DEV)).cpu().numpy()[0]
        for l in lams:
            out[f"reward_lam{l:g}"].append(heldout_r2(E[c], r, rng, l))
            out[f"reward_shuffled_lam{l:g}"].append(heldout_r2(E[c], rng.permutation(r), rng, l))
            out[f"ref_logprob_lam{l:g}"].append(heldout_r2(E[c], z["lp"][i], rng, l))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    dump("bottleneck", res)
    print(res)


@torch.no_grad()
def cross_r2(n_ctx=2000):
    """Same question for the features of the cross-feature head: a model fitted on half of the contexts predicts the
    within-context reward (r minus its context mean) on the other half -- how much of what pi* adds is expressible
    from "user x track" features plus item length/popularity/base log-prob."""
    import lightgbm as lgb
    from common import to_dev, vocab, windows
    from rm import cross_feats
    z = load_cache(1)
    idx, cand = z["idx"][:n_ctx], z["cand"][:n_ctx]
    lp_all, P_all, new_all = z["lp"][:n_ctx], z["P"][0, :n_ctx], z["new"][:n_ctx]
    voc = vocab()
    art_of = torch.as_tensor(voc["tok_artist"]).long().to(DEV)
    R = Reward("played", calibrate("rm0"))
    X, Y, G = [], [], []
    for i in range(0, n_ctx, 64):
        W = to_dev(windows(idx[i:i + 64]))
        c = torch.as_tensor(cand[i:i + 64]).long().to(DEV)
        allowed = W["mask"][:, None, :].expand(-1, c.shape[1], -1)
        q = torch.full(c.shape, float(W["tok"].shape[1]), device=DEV)
        cr = cross_feats(c, q, W["tok"].long(), W["played"].float(), art_of, allowed).cpu().numpy()
        cc = cand[i:i + 64]
        x = np.concatenate([cr, np.log(np.maximum(voc["tok_len"][cc], 10))[..., None],
                            np.log1p(voc["tok_pop"][cc])[..., None], lp_all[i:i + 64][..., None]], -1)
        r = R(torch.as_tensor(P_all[i:i + 64]).float().to(DEV), c, torch.as_tensor(new_all[i:i + 64]).to(DEV)).cpu().numpy()
        X.append(x)
        Y.append(r - r.mean(1, keepdims=True))
    X, Y = np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)
    h = n_ctx // 2
    Xtr, Ytr, Xte, Yte = X[:h].reshape(-1, X.shape[-1]), Y[:h].ravel(), X[h:].reshape(-1, X.shape[-1]), Y[h:].ravel()
    b = lgb.train(dict(objective="l2", learning_rate=0.1, num_leaves=63, verbose=-1), lgb.Dataset(Xtr, Ytr), 300)
    p = b.predict(Xte)
    r2 = 1 - ((Yte - p) ** 2).sum() / ((Yte - Yte.mean()) ** 2).sum()
    names = ["n_i", "m_i", "has_i", "n_a", "m_a", "rec_i", "same_last", "art_share", "loglen", "logpop", "lp_ref"]
    return {"cross_features_heldout_r2": float(r2),
            "gain": dict(zip(names, (b.feature_importance("gain") / b.feature_importance("gain").sum()).round(3).tolist()))}


if __name__ == "__main__":
    import sys
    if "--cross-only" not in sys.argv:
        main()
    import json
    from common import RES
    res = json.loads((RES / "bottleneck.json").read_text())
    res.update(cross_r2())
    dump("bottleneck", res)
    print(res)
